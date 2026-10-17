"""Walk-forward evaluation for ML models and trading strategies.

Examples:
    PYTHONPATH=src python -m marketmind.models
    PYTHONPATH=src python -m marketmind.models --tickers TSLA,AMD,GOOGL
    PYTHONPATH=src python -m marketmind.models --tickers NVDA,SPY --period 2y
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marketmind.backtest import run_backtest, run_swing_backtest, swing_positions
from marketmind.data import load_prices
from marketmind.features import build_features
from marketmind.metrics import calculate_metrics
from marketmind.strategies import RULE_STRATEGIES, generate_signals
from marketmind.tickers import parse_tickers


DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "SPY"]
TICKERS = DEFAULT_TICKERS
COST_PER_SIDE = 0.0005
REPORT_DIR = Path("reports")
INITIAL_TRAIN_FRACTION = 0.60
TEST_FOLD_FRACTION = 0.10

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_ratio",
    "price_vs_ma10",
    "price_vs_ma50",
    "volatility_10d",
    "volatility_20d",
    "volume_zscore",
    "intraday_range",
    "close_location",
]

MODELS = {
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1),
    ),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    ),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        random_state=42,
    ),
}


def _walk_forward_predictions(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """Create expanding-window predictions without training on test labels."""
    X = df[FEATURE_COLUMNS]
    y = (df["target_return"] > 0).astype(int)

    initial_train_size = int(len(df) * INITIAL_TRAIN_FRACTION)
    test_size = max(1, int(len(df) * TEST_FOLD_FRACTION))

    predictions = {
        name: pd.Series(np.nan, index=df.index, dtype=float)
        for name in MODELS
    }
    folds = pd.Series(np.nan, index=df.index, dtype=float)
    fold_number = 0

    for start in range(initial_train_size, len(df), test_size):
        end = min(start + test_size, len(df))

        # Leave the immediately preceding label out because it overlaps
        # the first test session's target return.
        X_train = X.iloc[: start - 1]
        y_train = y.iloc[: start - 1]
        X_test = X.iloc[start:end]

        if y_train.nunique() < 2:
            raise ValueError("Training window must contain both target classes.")

        for name, create_model in MODELS.items():
            model = create_model()
            model.fit(X_train, y_train)
            predictions[name].iloc[start:end] = model.predict(X_test)

        folds.iloc[start:end] = fold_number
        fold_number += 1

    if not any(series.notna().any() for series in predictions.values()):
        raise ValueError("Not enough data for walk-forward evaluation.")

    return predictions, folds, y


def _completed_trade_returns(daily: pd.DataFrame) -> list[float]:
    """Return compounded returns for completed, contiguous long trades."""
    if daily.empty or "signal" not in daily:
        return []

    positions = daily["signal"].fillna(0).astype(int).to_numpy()
    returns = daily["strategy_return"].fillna(0).astype(float).to_numpy()
    completed: list[float] = []
    i = 0

    while i < len(positions):
        if positions[i] != 1:
            i += 1
            continue

        start = i
        while i + 1 < len(positions) and positions[i + 1] == 1:
            i += 1

        end = i
        if end + 1 < len(positions) and positions[end + 1] == 0:
            completed.append(
                float((1 + returns[start : end + 1]).prod() - 1)
            )

        i += 1

    return completed


def _evaluate_signals(
    prices: pd.DataFrame,
    signals: pd.Series,
    ticker: str,
    strategy_name: str,
    test_signal_dates: pd.Index,
    prediction_accuracy: float = np.nan,
    balanced_accuracy: float = np.nan,
    folds: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate returns realized after out-of-sample signal dates.

    RSI and Bollinger Bands use swing returns (open-to-open). Other existing
    experiments retain their previous intraday methodology for continuity.
    """
    signals = signals.reindex(prices.index).fillna(0).astype(int)
    is_swing = strategy_name in {"RSI", "Bollinger Bands"}

    if is_swing:
        strategy_returns, _ = run_swing_backtest(
            prices,
            signals,
            cost_per_side=COST_PER_SIDE,
        )
        applied_positions = swing_positions(
            signals,
            liquidate_at_end=True,
        )
        benchmark = prices["Open"].shift(-1) / prices["Open"] - 1
        intraday_benchmark = pd.Series(
            np.nan,
            index=prices.index,
            dtype=float,
        )
    else:
        strategy_returns, _ = run_backtest(
            prices,
            signals,
            cost_per_side=COST_PER_SIDE,
        )
        applied_positions = signals.shift(1, fill_value=0).astype(int)
        benchmark = prices["Close"].pct_change().fillna(0)
        intraday = prices["Close"] / prices["Open"] - 1
        intraday_benchmark = intraday - 2 * COST_PER_SIDE

    signal_date = pd.Series(prices.index, index=prices.index).shift(1)

    daily = pd.DataFrame({
        "date": prices.index,
        "ticker": ticker,
        "strategy": strategy_name,
        "signal_date": signal_date.to_numpy(),
        "signal": applied_positions.to_numpy(),
        "strategy_return": strategy_returns.to_numpy(),
        "intraday_benchmark_return": intraday_benchmark.to_numpy(),
        "buy_hold_return": benchmark.to_numpy(),
    })

    # Keep the final no-return row temporarily as a position-state boundary.
    daily = daily[daily["signal_date"].isin(test_signal_dates)].copy()
    daily = daily.sort_values("date")
    daily_for_trade_stats = daily.copy()
    daily = daily[daily["strategy_return"].notna()].copy()

    if daily.empty:
        raise ValueError(
            f"No out-of-sample returns for {ticker} / {strategy_name}."
        )

    if is_swing:
        daily = daily.sort_values("date").copy()
        daily.iloc[0, daily.columns.get_loc("buy_hold_return")] -= COST_PER_SIDE
        daily.iloc[-1, daily.columns.get_loc("buy_hold_return")] -= COST_PER_SIDE

    strategy_metrics = calculate_metrics(daily["strategy_return"])
    buy_hold_metrics = calculate_metrics(daily["buy_hold_return"])

    intraday_metrics = (
        calculate_metrics(daily["intraday_benchmark_return"])
        if not is_swing
        else {"total_return": np.nan}
    )

    active_days = int((daily["signal"] == 1).sum())

    if is_swing:
        previous_position = daily["signal"].shift(1, fill_value=0)
        trade_count = int(
            ((daily["signal"] == 1) & (previous_position == 0)).sum()
        )
        trade_returns = _completed_trade_returns(daily_for_trade_stats)
        trade_win_rate = (
            float(np.mean(np.asarray(trade_returns) > 0))
            if trade_returns else np.nan
        )
        average_trade_return = (
            float(np.mean(trade_returns))
            if trade_returns else np.nan
        )
    else:
        trades = daily.loc[daily["signal"] == 1, "strategy_return"]
        trade_count = int(len(trades))
        trade_win_rate = (
            float((trades > 0).mean()) if trade_count else np.nan
        )
        average_trade_return = (
            float(trades.mean()) if trade_count else np.nan
        )

    summary = pd.DataFrame([{
        "ticker": ticker,
        "strategy": strategy_name,
        "backtest_type": (
            "swing_open_to_open"
            if is_swing else "intraday_open_to_close"
        ),
        "test_start": str(daily["date"].min().date()),
        "test_end": str(daily["date"].max().date()),
        "oos_days": len(daily),
        "active_days": active_days,
        "trade_count": trade_count,
        "completed_trade_win_rate": trade_win_rate,
        "average_trade_return": average_trade_return,
        "prediction_accuracy": prediction_accuracy,
        "balanced_accuracy": balanced_accuracy,
        **{
            f"strategy_{key}": value
            for key, value in strategy_metrics.items()
        },
        "intraday_benchmark_return": intraday_metrics["total_return"],
        "buy_hold_return": buy_hold_metrics["total_return"],
    }])

    return summary, daily


def evaluate_ticker(
    ticker: str,
    period: str = "5y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ML and rule strategies for one ticker over chosen history."""
    prices = load_prices(ticker=ticker, period=period)
    df = build_features(prices)

    predictions, folds, y = _walk_forward_predictions(df)
    test_dates = predictions[next(iter(MODELS))].dropna().index

    summaries: list[pd.DataFrame] = []
    daily_reports: list[pd.DataFrame] = []

    for model_name, predicted in predictions.items():
        valid = predicted.dropna().astype(int)
        actual = y.loc[valid.index]

        accuracy = float(accuracy_score(actual, valid))
        balanced = (
            float(balanced_accuracy_score(actual, valid))
            if actual.nunique() == 2
            else np.nan
        )

        summary, daily = _evaluate_signals(
            prices,
            predicted,
            ticker,
            f"ML: {model_name}",
            test_dates,
            accuracy,
            balanced,
            folds,
        )
        summaries.append(summary)
        daily_reports.append(daily)

    prediction_frame = pd.DataFrame(predictions)
    vote_count = prediction_frame.sum(axis=1, min_count=3)

    ensemble = (vote_count >= 2).astype(float).where(
        prediction_frame.notna().all(axis=1)
    )
    valid_ensemble = ensemble.dropna().astype(int)
    actual_ensemble = y.loc[valid_ensemble.index]

    ensemble_accuracy = float(
        accuracy_score(actual_ensemble, valid_ensemble)
    )
    ensemble_balanced = (
        float(balanced_accuracy_score(actual_ensemble, valid_ensemble))
        if actual_ensemble.nunique() == 2
        else np.nan
    )

    summary, daily = _evaluate_signals(
        prices,
        ensemble,
        ticker,
        "ML ensemble",
        test_dates,
        ensemble_accuracy,
        ensemble_balanced,
        folds,
    )
    summaries.append(summary)
    daily_reports.append(daily)

    for strategy_name in RULE_STRATEGIES:
        rule_signals = generate_signals(
            prices,
            strategy_name,
            start_date=test_dates[0],
        )
        rule_signals = rule_signals.where(
            rule_signals.index.isin(test_dates),
            0,
        )

        summary, daily = _evaluate_signals(
            prices,
            rule_signals,
            ticker,
            strategy_name,
            test_dates,
        )
        summaries.append(summary)
        daily_reports.append(daily)

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(daily_reports, ignore_index=True),
    )


def main(argv: list[str] | None = None) -> None:
    """Run reports for user-supplied Yahoo Finance ticker symbols."""
    parser = argparse.ArgumentParser(
        description="Run MarketMind strategy evaluations."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated Yahoo Finance symbols (default: NVDA,AAPL,MSFT,SPY).",
    )
    parser.add_argument(
        "--period",
        choices=["1y", "2y", "5y", "10y", "max"],
        default="5y",
        help="Historical price period to download (default: 5y).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR),
        help="Output directory for model_summary.csv and daily_returns.csv.",
    )
    args = parser.parse_args(argv)

    try:
        tickers = parse_tickers(args.tickers)
    except ValueError as exc:
        parser.error(str(exc))

    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    daily_reports: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            summary, daily = evaluate_ticker(ticker, period=args.period)
            summaries.append(summary)
            daily_reports.append(daily)
            print(f"Completed {ticker}: 9 strategies.")
        except Exception as exc:
            print(f"Failed {ticker}: {exc}")

    if not summaries:
        raise RuntimeError("No strategies completed successfully.")

    summary_report = pd.concat(summaries, ignore_index=True)
    daily_report = pd.concat(daily_reports, ignore_index=True)

    summary_report.to_csv(report_dir / "model_summary.csv", index=False)
    daily_report.to_csv(report_dir / "daily_returns.csv", index=False)

    columns = [
        "prediction_accuracy",
        "balanced_accuracy",
        "strategy_total_return",
        "strategy_annualized_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "strategy_annualized_volatility",
        "completed_trade_win_rate",
        "average_trade_return",
        "trade_count",
        "active_days",
        "intraday_benchmark_return",
        "buy_hold_return",
    ]

    print("\nAverage results by strategy:")
    print(
        summary_report.groupby("strategy")[columns]
        .mean()
        .round(4)
        .to_string()
    )
    print(f"\nSaved reports to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
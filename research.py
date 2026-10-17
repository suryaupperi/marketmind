"""Pure research helpers shared by the dashboard and automated tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketmind.backtest import run_swing_backtest, swing_positions
from marketmind.metrics import calculate_metrics


def chronological_test_start(
    index: pd.Index,
    train_fraction: float = 0.60,
    min_test_sessions: int = 10,
) -> pd.Timestamp:
    """Return the first holdout timestamp without using future target values."""
    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1"
        )

    if min_test_sessions < 1:
        raise ValueError("min_test_sessions must be at least 1")

    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(
            "Index must be unique and sorted in ascending time order"
        )

    split_at = int(len(index) * train_fraction)

    if len(index) - split_at < min_test_sessions or split_at < 1:
        raise ValueError(
            f"Need enough history for a holdout of at least "
            f"{min_test_sessions} sessions"
        )

    return pd.Timestamp(index[split_at])


def build_oos_daily(
    prices: pd.DataFrame,
    signals: pd.Series,
    ticker: str,
    strategy: str,
    test_start: pd.Timestamp,
    cost_per_side: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
    dict[str, float],
]:
    """Build aligned strategy/benchmark returns for one chronological holdout.

    Signals are generated after each close and acted on at the next open.
    The performance frame excludes the terminal row without a following
    open; the full frame retains it so end-of-window liquidations count as
    completed trades in trade-level analysis.
    """
    if not prices.index.equals(signals.index):
        raise ValueError("Prices and signals must have matching indexes")

    if test_start not in prices.index:
        raise ValueError(
            "test_start must be present in the price index"
        )

    if not np.isfinite(cost_per_side) or not 0 <= cost_per_side < 1:
        raise ValueError(
            "cost_per_side must be finite and between 0 and 1"
        )

    strategy_returns, _ = run_swing_backtest(
        prices,
        signals,
        cost_per_side=cost_per_side,
        liquidate_at_end=True,
    )

    positions = swing_positions(
        signals,
        liquidate_at_end=True,
    )

    open_to_open = (
        prices["Open"].shift(-1) / prices["Open"] - 1
    )

    signal_dates = pd.Series(
        prices.index,
        index=prices.index,
    ).shift(1)

    full = pd.DataFrame(
        {
            "date": prices.index,
            "ticker": ticker,
            "strategy": strategy,
            "signal_date": signal_dates.to_numpy(),
            "signal": positions.to_numpy(dtype=int),
            "strategy_return": strategy_returns.to_numpy(dtype=float),
            "buy_hold_return": open_to_open.to_numpy(dtype=float),
        },
        index=prices.index,
    )

    full = full.loc[
        full["signal_date"].notna()
        & (full["signal_date"] >= test_start)
    ].copy()

    valid = full.loc[
        full["strategy_return"].notna()
    ].copy()

    if valid.empty:
        raise ValueError("No usable out-of-sample return intervals")

    # Benchmark enters at the first tested open and exits at the final
    # tested interval's ending open. Both costs apply to one interval
    # if the holdout happens to contain only one interval.
    first_idx = valid.index[0]
    last_idx = valid.index[-1]

    full.loc[first_idx, "buy_hold_return"] -= cost_per_side
    full.loc[last_idx, "buy_hold_return"] -= cost_per_side

    # Re-select after updating the benchmark values in the full frame.
    valid = full.loc[valid.index].copy()

    strategy_metrics = calculate_metrics(
        valid["strategy_return"]
    )

    benchmark_metrics = calculate_metrics(
        valid["buy_hold_return"]
    )

    return (
        valid.reset_index(drop=True),
        full.reset_index(drop=True),
        strategy_metrics,
        benchmark_metrics,
    )
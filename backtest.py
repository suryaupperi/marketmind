"""Validated backtesting utilities for intraday and swing strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketmind.metrics import calculate_metrics


def _validate_inputs(
    prices: pd.DataFrame,
    signals: pd.Series,
    cost_per_side: float,
    required_price_columns: tuple[str, ...],
) -> None:
    if not np.isfinite(cost_per_side) or not 0 <= cost_per_side < 1:
        raise ValueError("cost_per_side must be finite and between 0 and 1")

    if not isinstance(prices, pd.DataFrame) or not isinstance(signals, pd.Series):
        raise TypeError("prices must be a DataFrame and signals must be a Series")

    if len(prices) != len(signals) or not prices.index.equals(signals.index):
        raise ValueError("Prices and signals must have matching indexes")

    if not prices.index.is_unique:
        raise ValueError("Price index must not contain duplicate timestamps")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be sorted in ascending time order")

    if signals.isna().any() or not signals.isin([0, 1]).all():
        raise ValueError(
            "Signals must contain only 0 (cash) or 1 (long), "
            "with no missing values"
        )

    missing = [
        column
        for column in required_price_columns
        if column not in prices.columns
    ]
    if missing:
        raise ValueError(
            f"Prices must include columns: {', '.join(missing)}"
        )

    if len(prices) < 2:
        raise ValueError("At least two price rows are required")

    for column in required_price_columns:
        values = pd.to_numeric(prices[column], errors="coerce")

        if values.isna().any() or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Price column {column!r} contains missing "
                "or non-finite values"
            )

        if (values <= 0).any():
            raise ValueError(
                f"Price column {column!r} must contain only positive values"
            )


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    cost_per_side: float = 0.0005,
) -> tuple[pd.Series, dict[str, float]]:
    _validate_inputs(
        prices, signals, cost_per_side, ("Open", "Close")
    )

    intraday_returns = (
        prices["Close"].astype(float)
        / prices["Open"].astype(float)
        - 1
    )

    positions = signals.shift(1, fill_value=0).astype(int)
    trading_costs = positions * (2 * cost_per_side)

    strategy_returns = (
        positions * intraday_returns - trading_costs
    ).rename("strategy_return")

    return strategy_returns, calculate_metrics(strategy_returns)


def swing_positions(
    signals: pd.Series,
    liquidate_at_end: bool = True,
) -> pd.Series:
    if signals.isna().any() or not signals.isin([0, 1]).all():
        raise ValueError(
            "Signals must contain only 0 (cash) or 1 (long), "
            "with no missing values"
        )

    positions = signals.shift(1, fill_value=0).astype(int)
    positions = positions.rename("position")

    if liquidate_at_end and len(positions):
        positions.iloc[-1] = 0

    return positions


def run_swing_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    cost_per_side: float = 0.0005,
    liquidate_at_end: bool = True,
) -> tuple[pd.Series, dict[str, float]]:
    _validate_inputs(prices, signals, cost_per_side, ("Open",))

    positions = swing_positions(
        signals,
        liquidate_at_end=liquidate_at_end,
    )

    open_prices = prices["Open"].astype(float)
    open_to_open_returns = open_prices.shift(-1) / open_prices - 1

    entries_at_open = (
        positions - positions.shift(1, fill_value=0)
    ).clip(lower=0)

    exits_at_next_open = (
        positions - positions.shift(-1, fill_value=0)
    ).clip(lower=0)

    costs = cost_per_side * (
        entries_at_open + exits_at_next_open
    )

    strategy_returns = (
        positions * open_to_open_returns - costs
    ).rename("strategy_return")

    return strategy_returns, calculate_metrics(strategy_returns)

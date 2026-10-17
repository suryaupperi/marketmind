"""Rule-based trading signals calculated from information available at each close."""

from collections.abc import Callable

import numpy as np
import pandas as pd

RULE_STRATEGIES = (
    "Moving-average crossover",
    "Momentum",
    "Mean reversion",
    "RSI",
    "Bollinger Bands",
)


def _position_from_entry_exit(
    index: pd.Index,
    entry: pd.Series,
    exit_: pd.Series,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Build long/cash signals, optionally starting flat on a test date."""
    entry_values = entry.reindex(index).fillna(False).to_numpy(dtype=bool)
    exit_values = exit_.reindex(index).fillna(False).to_numpy(dtype=bool)
    positions = np.zeros(len(index), dtype=int)

    start_at = 0
    if start_date is not None:
        locations = index.get_indexer([start_date])
        if len(locations) != 1 or locations[0] < 0:
            raise ValueError("start_date must be present in the price index")
        start_at = int(locations[0])

    # Begin flat at the test boundary. Historical prices still provide
    # rolling-indicator warm-up, but pre-test position state is discarded.
    active = False
    for i in range(start_at, len(index)):
        if active and exit_values[i]:
            active = False
        elif not active and entry_values[i]:
            active = True
        positions[i] = int(active)

    return pd.Series(positions, index=index, name="signal")


def moving_average_crossover(
    prices: pd.DataFrame, start_date: pd.Timestamp | None = None
) -> pd.Series:
    close = prices["Close"]
    fast = close.rolling(10, min_periods=10).mean()
    slow = close.rolling(50, min_periods=50).mean()
    return _position_from_entry_exit(
        prices.index, fast > slow, fast <= slow, start_date
    )


def momentum(
    prices: pd.DataFrame,
    entry_threshold: float = 0.02,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    ret20 = prices["Close"].pct_change(20)
    return _position_from_entry_exit(
        prices.index, ret20 > entry_threshold, ret20 <= 0, start_date
    )


def mean_reversion(
    prices: pd.DataFrame,
    entry_z: float = -1.5,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    close = prices["Close"]
    average = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    zscore = (close - average) / std
    return _position_from_entry_exit(
        prices.index, zscore < entry_z, zscore >= 0, start_date
    )


def rsi_strategy(
    prices: pd.DataFrame, start_date: pd.Timestamp | None = None
) -> pd.Series:
    change = prices["Close"].diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(loss.ne(0), 100.0)
    return _position_from_entry_exit(
        prices.index, rsi < 30, rsi > 50, start_date
    )


def bollinger_bands(
    prices: pd.DataFrame, start_date: pd.Timestamp | None = None
) -> pd.Series:
    close = prices["Close"]
    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    lower = middle - 2 * std
    crossed_back_above_lower = (
        (close > lower) & (close.shift(1) <= lower.shift(1))
    )
    return _position_from_entry_exit(
        prices.index, crossed_back_above_lower, close >= middle, start_date
    )


_STRATEGY_FUNCTIONS: dict[str, Callable[..., pd.Series]] = {
    "Moving-average crossover": moving_average_crossover,
    "Momentum": momentum,
    "Mean reversion": mean_reversion,
    "RSI": rsi_strategy,
    "Bollinger Bands": bollinger_bands,
}


def generate_signals(
    prices: pd.DataFrame,
    strategy_name: str,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Generate aligned 0/1 signals, optionally resetting at start_date."""
    try:
        strategy = _STRATEGY_FUNCTIONS[strategy_name]
    except KeyError as exc:
        choices = ", ".join(RULE_STRATEGIES)
        raise ValueError(
            f"Unknown strategy {strategy_name!r}. Choose from: {choices}"
        ) from exc

    signals = strategy(prices, start_date=start_date)

    if not signals.index.equals(prices.index) or not signals.isin([0, 1]).all():
        raise ValueError("Strategy must return aligned 0/1 signals")

    return signals.astype(int).rename("signal")
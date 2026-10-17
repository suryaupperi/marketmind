
import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy().sort_index()

    close = df["Close"]
    volume = df["Volume"]

    # Historical returns
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # Trend and momentum
    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    df["ma_ratio"] = ma_20 / ma_50 - 1
    df["price_vs_ma10"] = close / ma_10 - 1
    df["price_vs_ma50"] = close / ma_50 - 1

    # Historical volatility
    daily_returns = df["return_1d"]
    df["volatility_10d"] = daily_returns.rolling(10).std()
    df["volatility_20d"] = daily_returns.rolling(20).std()

    # Volume behavior
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - volume_mean) / volume_std

    # Daily range and close location within the range
    df["intraday_range"] = (df["High"] - df["Low"]) / close
    daily_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_location"] = (
        (close - df["Low"]) / daily_range
    )

    # Next-day close-to-close return, for prediction experiments.
    df["target_return"] = (df["Close"] / df["Open"] - 1).shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

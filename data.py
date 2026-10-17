
import yfinance as yf
import pandas as pd


def load_prices(
    ticker: str = "NVDA",
    period: str = "5y",
) -> pd.DataFrame:
    prices = yf.Ticker(ticker).history(
        period=period,
        auto_adjust=True,
    )

    if prices.empty:
        raise ValueError(f"No market data returned for {ticker}")

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    required = ["Open", "High", "Low", "Close", "Volume"]
    prices = prices.dropna(subset=required)

    if (prices["Close"] <= 0).any():
        raise ValueError("Prices must be positive")

    return prices


if __name__ == "__main__":
    df = load_prices()
    print(df.tail())
    print(f"Rows downloaded: {len(df)}")

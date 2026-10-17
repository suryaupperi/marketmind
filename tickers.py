"""Shared parsing and validation for user-provided ticker symbols."""

import re

MAX_TICKERS = 10


def parse_tickers(
    raw: str,
    max_tickers: int = MAX_TICKERS,
) -> list[str]:
    """Parse comma-separated Yahoo Finance symbols, preserving input order.

    Examples: AAPL, TSLA, BRK-B, ^GSPC, BTC-USD, EURUSD=X.
    Duplicate symbols are removed case-insensitively.
    """
    symbols: list[str] = []

    for part in raw.split(","):
        symbol = part.strip().upper()

        if not symbol:
            continue

        if not re.fullmatch(r"[A-Z0-9.^=_-]{1,20}", symbol):
            raise ValueError(
                f"Ticker '{symbol}' contains unsupported characters. "
                "Enter symbols separated by commas, such as AAPL, TSLA, or BRK-B."
            )

        if symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise ValueError("Enter at least one ticker symbol.")

    if len(symbols) > max_tickers:
        raise ValueError(
            f"Please enter no more than {max_tickers} tickers at once."
        )

    return symbols
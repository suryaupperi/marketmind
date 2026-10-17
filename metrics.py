"""Performance metrics for a chronological series of strategy returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate total/annualized return, Sharpe, volatility, and drawdown.

    Returns are assumed to be one observation per trading session.
    Drawdown is measured from initial equity of 1.0, so an initial loss
    is not overlooked. Missing terminal returns are ignored; non-finite
    returns are rejected.
    """
    try:
        values = returns.astype(float).dropna()
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "returns must be a numeric pandas Series"
        ) from exc

    if values.empty:
        raise ValueError("Returns cannot be empty")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Returns must be finite or missing")

    if (values < -1).any():
        raise ValueError("Returns cannot be less than -100%")

    equity = (1 + values).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = len(values) / 252

    if equity.iloc[-1] <= 0:
        annualized_return = -1.0
    else:
        annualized_return = float(
            equity.iloc[-1] ** (1 / years) - 1
        )

    daily_volatility = values.std(ddof=1)

    sharpe = (
        float(
            values.mean()
            / daily_volatility
            * np.sqrt(252)
        )
        if pd.notna(daily_volatility) and daily_volatility > 0
        else 0.0
    )

    # Include initial equity of 1.0 so an initial loss counts as drawdown.
    equity_path = np.concatenate(
        ([1.0], equity.to_numpy(dtype=float))
    )

    drawdown = (
        equity_path / np.maximum.accumulate(equity_path) - 1
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": (
            float(daily_volatility * np.sqrt(252))
            if pd.notna(daily_volatility)
            else 0.0
        ),
    }
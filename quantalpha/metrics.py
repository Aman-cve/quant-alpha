"""Performance and risk metrics for a daily return stream.

Everything here operates on a 1-D numpy array of simple daily returns and uses
the standard 252-trading-day annualisation convention.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def equity_curve(returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Cumulative growth of 1 unit of capital."""
    return start * np.cumprod(1.0 + returns)


def total_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def cagr(returns: np.ndarray) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / n) - 1.0


def ann_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def sharpe(returns: np.ndarray, ann_rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    rf_daily = ann_rf / TRADING_DAYS
    excess = returns - rf_daily
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: np.ndarray, ann_rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    rf_daily = ann_rf / TRADING_DAYS
    excess = returns - rf_daily
    downside = excess[excess < 0]
    dd = np.sqrt(np.mean(downside ** 2)) if len(downside) else 0.0
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough decline, returned as a negative fraction."""
    if len(returns) == 0:
        return 0.0
    curve = equity_curve(returns)
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    return float(dd.min())


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    curve = equity_curve(returns)
    peak = np.maximum.accumulate(curve)
    return curve / peak - 1.0


def calmar(returns: np.ndarray) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return cagr(returns) / abs(mdd)


def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def summary(returns: np.ndarray, ann_rf: float = 0.0, turnover: float | None = None) -> dict:
    """A full tearsheet dict for one strategy."""
    out = {
        "Total Return": total_return(returns),
        "CAGR": cagr(returns),
        "Volatility": ann_volatility(returns),
        "Sharpe": sharpe(returns, ann_rf),
        "Sortino": sortino(returns, ann_rf),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar(returns),
        "Win Rate": win_rate(returns),
    }
    if turnover is not None:
        out["Avg Daily Turnover"] = float(turnover)
    return out

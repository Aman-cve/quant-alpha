"""Shared helpers for strategies.

Every strategy is a callable ``prices -> weights`` where ``prices`` is a
``(T, N)`` close-price panel and ``weights`` is a ``(T, N)`` array of target
portfolio weights.  ``weights[t]`` must be computable from ``prices[:t+1]``
only -- no peeking at the future.  The backtester applies ``weights[t]`` to the
return earned from day ``t`` to ``t+1``.
"""
from __future__ import annotations

import numpy as np


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean along axis 0; NaN during the warm-up period."""
    T = x.shape[0]
    out = np.full_like(x, np.nan, dtype=float)
    csum = np.cumsum(x, axis=0)
    for t in range(window - 1, T):
        if t == window - 1:
            out[t] = csum[t] / window
        else:
            out[t] = (csum[t] - csum[t - window]) / window
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    T = x.shape[0]
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(window - 1, T):
        out[t] = np.std(x[t - window + 1:t + 1], axis=0, ddof=1)
    return out


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    return rolling_mean(prices, window)


def past_return(prices: np.ndarray, lookback: int, skip: int = 0) -> np.ndarray:
    """Return over ``[t-skip-lookback, t-skip]``; NaN until enough history."""
    T, N = prices.shape
    out = np.full((T, N), np.nan, dtype=float)
    for t in range(lookback + skip, T):
        out[t] = prices[t - skip] / prices[t - skip - lookback] - 1.0
    return out


def dollar_neutral(signal_row: np.ndarray) -> np.ndarray:
    """Turn a cross-sectional signal into dollar-neutral, gross=1 weights."""
    s = signal_row - np.nanmean(signal_row)
    gross = np.nansum(np.abs(s))
    if gross == 0 or not np.isfinite(gross):
        return np.zeros_like(signal_row)
    return s / gross


def hold_forward(weights: np.ndarray, rebalance_days: set[int]) -> np.ndarray:
    """Carry weights forward, only changing them on rebalance days."""
    T = weights.shape[0]
    out = np.zeros_like(weights)
    cur = np.zeros(weights.shape[1])
    for t in range(T):
        if t in rebalance_days:
            cur = weights[t]
        out[t] = cur
    return out

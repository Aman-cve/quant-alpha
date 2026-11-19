"""Time-series trend following.

A long-only CTA-style rule: hold an equal-weight basket of every name trading
above its 200-day moving average, step aside (to cash) on the ones that aren't.
Captures sustained up-trends while sidestepping the worst drawdowns.
"""
from __future__ import annotations

import numpy as np

from .base import sma


def trend(prices: np.ndarray, window: int = 200) -> np.ndarray:
    T, N = prices.shape
    ma = sma(prices, window)
    weights = np.zeros((T, N))
    for t in range(T):
        if np.any(np.isnan(ma[t])):
            continue
        in_trend = prices[t] > ma[t]
        k = int(np.sum(in_trend))
        if k:
            weights[t, in_trend] = 1.0 / k   # equal-weight, fully invested
    return weights

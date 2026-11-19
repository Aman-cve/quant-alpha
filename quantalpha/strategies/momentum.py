"""Cross-sectional momentum.

The classic hedge-fund factor: go long recent winners, short recent losers.
We use a 6-month formation window skipping the most recent month (the "6-1"
convention, which sidesteps short-term reversal), rebalanced monthly.
"""
from __future__ import annotations

import numpy as np

from .base import hold_forward, past_return


def momentum(prices: np.ndarray, lookback: int = 126, skip: int = 21,
             n_side: int = 3, rebalance: int = 21) -> np.ndarray:
    T, N = prices.shape
    n_side = min(n_side, N // 2) or 1
    mom = past_return(prices, lookback, skip)

    raw = np.zeros((T, N))
    rebal_days = set()
    for t in range(T):
        row = mom[t]
        if np.any(np.isnan(row)):
            continue
        rebal_days.add(t) if (t % rebalance == 0) else None
        order = np.argsort(row)             # ascending: losers first
        longs, shorts = order[-n_side:], order[:n_side]
        w = np.zeros(N)
        w[longs] = 0.5 / n_side
        w[shorts] = -0.5 / n_side
        raw[t] = w
    if not rebal_days:                       # not enough history anywhere
        return np.zeros((T, N))
    return hold_forward(raw, rebal_days)

"""Short-term cross-sectional mean reversion.

Bet that the past week's relative losers bounce and winners cool off -- a
market-neutral book rebalanced daily.  Signal is the negative of the 5-day
return, demeaned across the universe and scaled to gross exposure of 1.
"""
from __future__ import annotations

import numpy as np

from .base import dollar_neutral, past_return


def mean_reversion(prices: np.ndarray, lookback: int = 5) -> np.ndarray:
    T, N = prices.shape
    r = past_return(prices, lookback, skip=0)
    weights = np.zeros((T, N))
    for t in range(T):
        row = r[t]
        if np.any(np.isnan(row)):
            continue
        weights[t] = dollar_neutral(-row)   # long losers, short winners
    return weights

"""Statistical-arbitrage pairs trading.

Finds the most correlated pair over an initial formation window, then trades the
mean-reverting *spread* between them.  A rolling OLS gives the hedge ratio; the
spread's z-score drives a market-neutral long/short with entry/exit hysteresis:

    z >  +entry  -> spread rich  -> short A / long B
    z <  -entry  -> spread cheap -> long A / short B
    |z| <  exit  -> flat

Pair selection uses only the formation window, so the traded period is out of
sample.
"""
from __future__ import annotations

import numpy as np


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return 0.0
    return float(np.dot(xc, y - y.mean()) / denom)


def _best_pair(prices: np.ndarray, formation: int) -> tuple[int, int]:
    rets = prices[1:formation] / prices[:formation - 1] - 1.0
    N = prices.shape[1]
    best, best_c = (0, 1), -2.0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = rets[:, i], rets[:, j]
            sd = a.std() * b.std()
            if sd == 0:
                continue
            c = float(np.mean((a - a.mean()) * (b - b.mean())) / sd)
            if c > best_c:
                best_c, best = c, (i, j)
    return best


def pairs(prices: np.ndarray, formation: int = 120, beta_win: int = 60,
          z_win: int = 20, entry: float = 1.0, exit_: float = 0.25):
    T, N = prices.shape
    weights = np.zeros((T, N))
    if T <= formation + beta_win:
        return weights
    a, b = _best_pair(prices, formation)

    state = 0  # +1 long spread (long A/short B), -1 short spread, 0 flat
    for t in range(formation, T):
        pa = prices[t - beta_win + 1:t + 1, a]
        pb = prices[t - beta_win + 1:t + 1, b]
        beta = _ols_beta(pa, pb)
        spread = pa - beta * pb
        sd = spread[-z_win:].std(ddof=1)
        if sd == 0:
            weights[t, a], weights[t, b] = _legs(state)
            continue
        z = (spread[-1] - spread[-z_win:].mean()) / sd
        if z > entry:
            state = -1
        elif z < -entry:
            state = +1
        elif abs(z) < exit_:
            state = 0
        weights[t, a], weights[t, b] = _legs(state)
    return weights


def _legs(state: int) -> tuple[float, float]:
    """Dollar-neutral, gross=1 legs for (A, B) given the spread state."""
    if state == +1:      # long spread: long A, short B
        return 0.5, -0.5
    if state == -1:      # short spread: short A, long B
        return -0.5, 0.5
    return 0.0, 0.0

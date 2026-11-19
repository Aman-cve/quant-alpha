"""Vectorised backtester with transaction costs.

Given a close-price panel and a strategy's target weights, it computes the net
daily return stream.  Positions decided at the close of day ``t`` earn the
return from ``t`` to ``t+1``; rebalancing from one day's weights to the next
incurs a cost of ``cost_bps`` on the traded notional (turnover).
"""
from __future__ import annotations

import numpy as np


class BacktestResult:
    def __init__(self, dates, net, gross, turnover):
        self.dates = dates          # aligned with net (len T-1)
        self.net = net              # net daily returns
        self.gross = gross          # before costs
        self.turnover = turnover    # per-day traded notional

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover)) if len(self.turnover) else 0.0


def run(prices: np.ndarray, weights: np.ndarray, dates: list[str],
        cost_bps: float = 5.0) -> BacktestResult:
    if prices.shape != weights.shape:
        raise ValueError("prices and weights must share shape")
    asset_ret = prices[1:] / prices[:-1] - 1.0           # (T-1, N), day t->t+1
    pos = weights[:-1]                                    # held across that day
    gross = np.sum(pos * asset_ret, axis=1)              # (T-1,)
    traded = np.abs(np.diff(weights, axis=0))            # rebalance each day
    turnover = np.sum(traded, axis=1)                    # (T-1,)
    cost = (cost_bps / 1e4) * turnover
    net = gross - cost
    return BacktestResult(dates[1:], net, gross, turnover)


def buy_and_hold(prices_1d: np.ndarray, dates: list[str]) -> BacktestResult:
    """Benchmark: hold a single series (e.g. SPY / Nifty) through the window."""
    ret = prices_1d[1:] / prices_1d[:-1] - 1.0
    zeros = np.zeros_like(ret)
    return BacktestResult(dates[1:], ret, ret, zeros)

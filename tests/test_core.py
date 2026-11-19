"""Unit tests for metrics, the backtester, and strategy no-lookahead safety."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantalpha import backtest, charts, metrics  # noqa: E402
from quantalpha.strategies import REGISTRY, momentum  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_equity_curve_compounds(self):
        r = np.array([0.1, -0.1, 0.05])
        curve = metrics.equity_curve(r)
        self.assertAlmostEqual(curve[-1], 1.1 * 0.9 * 1.05)

    def test_zero_returns_are_neutral(self):
        r = np.zeros(252)
        self.assertEqual(metrics.cagr(r), 0.0)
        self.assertEqual(metrics.sharpe(r), 0.0)
        self.assertEqual(metrics.max_drawdown(r), 0.0)

    def test_cagr_annualises(self):
        daily = 0.001
        r = np.full(252, daily)
        self.assertAlmostEqual(metrics.cagr(r), (1 + daily) ** 252 - 1, places=6)

    def test_max_drawdown_is_negative(self):
        r = np.array([0.2, -0.5, 0.1])
        self.assertLess(metrics.max_drawdown(r), 0.0)

    def test_sharpe_sign(self):
        up = np.full(252, 0.001)
        self.assertGreater(metrics.sharpe(up), 0)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.dates = [f"2024-01-{i+1:02d}" for i in range(5)]
        self.prices = np.array([
            [100.0, 50.0], [101.0, 49.0], [102.0, 50.0],
            [101.0, 51.0], [103.0, 52.0],
        ])

    def test_shapes_align(self):
        w = np.ones_like(self.prices) * 0.5
        res = backtest.run(self.prices, w, self.dates)
        self.assertEqual(len(res.net), len(self.prices) - 1)
        self.assertEqual(len(res.dates), len(res.net))

    def test_costs_reduce_return(self):
        # alternate full long between the two names -> churns, so costs bite
        w = np.zeros_like(self.prices)
        w[::2, 0] = 1.0
        w[1::2, 1] = 1.0
        free = backtest.run(self.prices, w, self.dates, cost_bps=0.0)
        charged = backtest.run(self.prices, w, self.dates, cost_bps=50.0)
        self.assertLess(charged.net.sum(), free.net.sum())

    def test_buy_and_hold_matches_price_return(self):
        px = self.prices[:, 0]
        res = backtest.buy_and_hold(px, self.dates)
        self.assertAlmostEqual(res.net[-1], px[-1] / px[-2] - 1)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            backtest.run(self.prices, np.ones((3, 2)), self.dates)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        steps = rng.normal(0.0005, 0.01, size=(400, 6))
        self.prices = 100 * np.cumprod(1 + steps, axis=0)

    def test_all_strategies_return_valid_weights(self):
        for name, fn in REGISTRY.items():
            w = fn(self.prices)
            self.assertEqual(w.shape, self.prices.shape, name)
            self.assertFalse(np.any(np.isnan(w)), name)
            # gross exposure never exceeds ~1 (within rounding)
            self.assertTrue(np.all(np.sum(np.abs(w), axis=1) <= 1.0 + 1e-9), name)

    def test_no_lookahead(self):
        # weights at time t must not change if future prices change
        w_full = momentum(self.prices)
        cut = 300
        truncated = self.prices.copy()
        truncated[cut:] += 20.0  # perturb the future
        w_trunc = momentum(truncated)
        np.testing.assert_allclose(w_full[:cut], w_trunc[:cut], atol=1e-12)


class TestCharts(unittest.TestCase):
    def test_equity_chart_is_svg(self):
        svg = charts.equity_chart({"A": metrics.equity_curve(np.full(50, 0.001))})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)

    def test_drawdown_chart_is_svg(self):
        svg = charts.drawdown_chart(metrics.drawdown_series(np.full(50, -0.001)))
        self.assertTrue(svg.startswith("<svg"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

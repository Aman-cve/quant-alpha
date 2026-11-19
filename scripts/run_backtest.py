"""Run every strategy on the US and Indian universes, print tearsheets, and
write SVG report charts + a markdown results table.

    python3 scripts/run_backtest.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantalpha import backtest, charts, data, metrics, universe  # noqa: E402
from quantalpha.strategies import REGISTRY  # noqa: E402

COST_BPS = 5.0
REPORTS = ROOT / "reports"


def run_market(code: str) -> dict:
    cfg = universe.MARKETS[code]
    tickers, bench_sym, rf = cfg["tickers"], cfg["benchmark"], cfg["ann_rf"]
    dates, prices_all = data.load_panel(tickers + [bench_sym])
    prices = prices_all[:, :len(tickers)]
    bench = prices_all[:, len(tickers)]

    results, rows = {}, {}
    strat_returns = []
    for name, fn in REGISTRY.items():
        weights = fn(prices)
        res = backtest.run(prices, weights, dates, cost_bps=COST_BPS)
        results[name] = res
        strat_returns.append(res.net)
        rows[name] = metrics.summary(res.net, ann_rf=rf, turnover=res.avg_turnover)

    # equal-weight multi-strategy book
    combo = np.mean(np.vstack(strat_returns), axis=0)
    rows["Multi-Strat Book"] = metrics.summary(combo, ann_rf=rf)

    bench_res = backtest.buy_and_hold(bench, dates)
    rows[f"Benchmark ({bench_sym})"] = metrics.summary(bench_res.net, ann_rf=rf)

    _print_table(code, rows)
    _write_charts(code, results, combo, bench_res, bench_sym)
    return rows


def _print_table(code: str, rows: dict) -> None:
    cols = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Win Rate"]
    print(f"\n=== {code} market — {len(rows)} books (costs {COST_BPS:.0f} bps) ===")
    head = f"{'Strategy':26}" + "".join(f"{c:>13}" for c in cols)
    print(head)
    print("-" * len(head))
    for name, m in rows.items():
        line = f"{name:26}"
        for c in cols:
            v = m[c]
            if c in ("CAGR", "Volatility", "Max Drawdown", "Win Rate"):
                line += f"{v*100:>12.1f}%"
            else:
                line += f"{v:>13.2f}"
        print(line)


def _write_charts(code, results, combo, bench_res, bench_sym) -> None:
    REPORTS.mkdir(exist_ok=True)
    curves = {name: metrics.equity_curve(r.net) for name, r in results.items()}
    curves["Multi-Strat Book"] = metrics.equity_curve(combo)
    curves[f"Benchmark ({bench_sym})"] = metrics.equity_curve(bench_res.net)
    title = "US Equities" if code == "US" else "Indian Equities (NSE)"
    svg = charts.equity_chart(curves, title=f"{title} — Growth of $1  (net of costs)")
    (REPORTS / f"{code.lower()}_equity.svg").write_text(svg)

    dd = metrics.drawdown_series(combo)
    svg_dd = charts.drawdown_chart(dd, title=f"{title} — Multi-Strat Drawdown")
    (REPORTS / f"{code.lower()}_drawdown.svg").write_text(svg_dd)


def main() -> None:
    all_rows = {}
    for code in ("US", "IN"):
        all_rows[code] = run_market(code)
    (REPORTS / "summary.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote charts + summary to {REPORTS}/")


if __name__ == "__main__":
    main()

"""Market data loading.

Pulls daily OHLCV from the public Yahoo Finance chart endpoint using nothing
but the Python standard library, then caches each symbol to a CSV under
``data/cache/``.  Once cached, backtests run fully offline -- handy on a slow
connection and so the committed repo is reproducible without a network call.
"""
from __future__ import annotations

import csv
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


class Series:
    """A single symbol's aligned date/close history."""

    def __init__(self, symbol: str, dates: list[str], close: np.ndarray):
        self.symbol = symbol
        self.dates = dates
        self.close = close

    def __len__(self) -> int:
        return len(self.dates)

    def returns(self) -> np.ndarray:
        """Simple daily returns, length N-1."""
        c = self.close
        return c[1:] / c[:-1] - 1.0


def _fetch_yahoo(symbol: str, rng: str = "2y", retries: int = 4) -> tuple[list[str], np.ndarray]:
    url = CHART_URL.format(sym=urllib.parse.quote(symbol), rng=rng)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            socket.setdefaulttimeout(25)
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            raw = urllib.request.urlopen(req).read().decode()
            res = json.loads(raw)["chart"]["result"][0]
            stamps = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
            dates, vals = [], []
            for ts, c in zip(stamps, closes):
                if c is None:
                    continue
                dates.append(time.strftime("%Y-%m-%d", time.gmtime(ts)))
                vals.append(float(c))
            if len(vals) < 30:
                raise ValueError(f"too few rows ({len(vals)}) for {symbol}")
            return dates, np.asarray(vals, dtype=float)
        except (urllib.error.URLError, socket.timeout, ValueError, KeyError, OSError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))  # back off, the first hit is often slow
    raise RuntimeError(f"failed to fetch {symbol}: {last_err}")


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.csv"


def _write_cache(symbol: str, dates: list[str], close: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_path(symbol).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "close"])
        for d, c in zip(dates, close):
            w.writerow([d, f"{c:.6f}"])


def _read_cache(symbol: str) -> tuple[list[str], np.ndarray]:
    dates, vals = [], []
    with _cache_path(symbol).open(newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            dates.append(row[0])
            vals.append(float(row[1]))
    return dates, np.asarray(vals, dtype=float)


def load(symbol: str, rng: str = "2y", refresh: bool = False) -> Series:
    """Load one symbol, from cache if present, otherwise fetch and cache it."""
    if not refresh and _cache_path(symbol).exists():
        dates, close = _read_cache(symbol)
    else:
        dates, close = _fetch_yahoo(symbol, rng=rng)
        _write_cache(symbol, dates, close)
    return Series(symbol, dates, close)


def load_panel(symbols: list[str], rng: str = "2y", refresh: bool = False):
    """Load many symbols and align them on their common trading dates.

    Returns ``(dates, prices)`` where ``prices`` is an ``(T, N)`` array of
    close prices, columns ordered like ``symbols``.
    """
    series = {s: load(s, rng=rng, refresh=refresh) for s in symbols}
    common = set(series[symbols[0]].dates)
    for s in symbols[1:]:
        common &= set(series[s].dates)
    dates = sorted(common)
    idx = {s: {d: i for i, d in enumerate(series[s].dates)} for s in symbols}
    prices = np.empty((len(dates), len(symbols)), dtype=float)
    for j, s in enumerate(symbols):
        col = series[s].close
        m = idx[s]
        prices[:, j] = [col[m[d]] for d in dates]
    return dates, prices

"""Tradable universes and benchmarks for US and Indian equities."""

# US large caps + S&P 500 ETF benchmark
US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "V"]
US_BENCHMARK = "SPY"

# NSE large caps (Yahoo ".NS" suffix) + Nifty 50 index benchmark
INDIA = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
]
INDIA_BENCHMARK = "^NSEI"

MARKETS = {
    "US": {"tickers": US, "benchmark": US_BENCHMARK, "ann_rf": 0.045},
    "IN": {"tickers": INDIA, "benchmark": INDIA_BENCHMARK, "ann_rf": 0.065},
}


def all_symbols() -> list[str]:
    syms = set(US + [US_BENCHMARK] + INDIA + [INDIA_BENCHMARK])
    return sorted(syms)

"""Strategy library. Each is a callable ``prices -> weights``."""

from .mean_reversion import mean_reversion
from .momentum import momentum
from .pairs import pairs
from .trend import trend

REGISTRY = {
    "Momentum (6-1, L/S)": momentum,
    "Mean Reversion (5d, L/S)": mean_reversion,
    "Trend Following (200d)": trend,
    "Stat-Arb Pairs": pairs,
}

__all__ = ["momentum", "mean_reversion", "trend", "pairs", "REGISTRY"]

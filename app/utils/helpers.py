"""Small shared utilities used across the data/technical/agent layers."""
from __future__ import annotations

from typing import List, Optional, Sequence


def pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """Percentage change from old -> new. Returns None if not computable."""
    if old in (None, 0) or new is None:
        return None
    try:
        return round(((new - old) / abs(old)) * 100, 2)
    except (TypeError, ZeroDivisionError):
        return None


def cagr(series: Sequence[float], periods: Optional[int] = None) -> Optional[float]:
    """Compound annual growth rate over a series (oldest -> newest)."""
    series = [v for v in series if v is not None]
    if len(series) < 2 or series[0] <= 0:
        return None
    n = periods or (len(series) - 1)
    if n <= 0:
        return None
    try:
        return round(((series[-1] / series[0]) ** (1 / n) - 1) * 100, 2)
    except (ZeroDivisionError, ValueError):
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return a / b


def latest_growth(series: Sequence[float]) -> Optional[float]:
    """Growth of the most recent period vs the one before it."""
    series = [v for v in series if v is not None]
    if len(series) < 2:
        return None
    return pct_change(series[-2], series[-1])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_levels(levels: List[float], ndigits: int = 2) -> List[float]:
    return sorted({round(l, ndigits) for l in levels})

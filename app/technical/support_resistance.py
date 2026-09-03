"""
Support / resistance level detection using clustered local extrema,
plus simple higher-high/higher-low structure detection.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from app.models.schemas import SupportResistance
from app.utils.helpers import round_levels


def _cluster_levels(levels: List[float], tolerance: float = 0.015) -> List[float]:
    """Merge nearby price levels (within `tolerance` fraction) into one,
    averaging the cluster so we don't return 10 near-duplicate lines."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters: List[List[float]] = [[levels[0]]]

    for lvl in levels[1:]:
        if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return [round(sum(c) / len(c), 2) for c in clusters]


def find_support_resistance(df: pd.DataFrame, window: int = 120, order: int = 5, max_levels: int = 4) -> SupportResistance:
    sub = df.tail(window)
    close = sub["close"].to_numpy()

    if len(close) < order * 2 + 1:
        last = float(close[-1]) if len(close) else 0.0
        return SupportResistance(support=[last * 0.97], resistance=[last * 1.03])

    maxima_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    minima_idx = argrelextrema(close, np.less_equal, order=order)[0]

    resistance_raw = close[maxima_idx].tolist()
    support_raw = close[minima_idx].tolist()

    resistance = _cluster_levels(resistance_raw)
    support = _cluster_levels(support_raw)

    current_price = close[-1]
    resistance = sorted([r for r in resistance if r >= current_price * 0.995])[:max_levels]
    support = sorted([s for s in support if s <= current_price * 1.005], reverse=True)[:max_levels]

    if not resistance:
        resistance = [round(current_price * 1.05, 2)]
    if not support:
        support = [round(current_price * 0.95, 2)]

    return SupportResistance(support=round_levels(support), resistance=round_levels(resistance))


def price_structure(df: pd.DataFrame, window: int = 60, order: int = 4) -> str:
    """Returns one of: 'higher_high_higher_low', 'lower_high_lower_low', 'mixed'."""
    sub = df.tail(window)
    close = sub["close"].to_numpy()

    maxima_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    minima_idx = argrelextrema(close, np.less_equal, order=order)[0]

    if len(maxima_idx) < 2 or len(minima_idx) < 2:
        return "mixed"

    highs_rising = close[maxima_idx[-1]] > close[maxima_idx[-2]]
    lows_rising = close[minima_idx[-1]] > close[minima_idx[-2]]

    if highs_rising and lows_rising:
        return "higher_high_higher_low"
    if not highs_rising and not lows_rising:
        return "lower_high_lower_low"
    return "mixed"

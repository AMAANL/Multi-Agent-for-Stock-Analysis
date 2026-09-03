"""
Multi-candle chart pattern detection (Cup & Handle, Double Top/Bottom,
Head & Shoulders, triangles, flag/pennant, rectangle).

These look at a rolling window rather than 2-3 candles. Detection is
still fully deterministic — local extrema + geometric rules — so it
stays testable and doesn't rely on an LLM "eyeballing" a chart image.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from app.models.schemas import ChartPattern


def _local_extrema(close: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of local maxima and minima using a symmetric window."""
    maxima = argrelextrema(close, np.greater_equal, order=order)[0]
    minima = argrelextrema(close, np.less_equal, order=order)[0]
    # de-duplicate consecutive equal-value plateaus
    maxima = np.array(sorted(set(maxima.tolist())))
    minima = np.array(sorted(set(minima.tolist())))
    return maxima, minima


def cup_and_handle(df: pd.DataFrame, window: int = 90) -> ChartPattern:
    """
    Left resistance -> decline -> rounded bottom -> recovery to near
    previous resistance -> small pullback (handle) -> breakout level.
    """
    sub = df.tail(window)
    if len(sub) < 40:
        return ChartPattern(name="Cup and Handle", detected=False, confidence=0.0)

    close = sub["close"].to_numpy()
    n = len(close)

    left_idx = int(np.argmax(close[: n // 3]))
    left_peak = close[left_idx]

    mid_start, mid_end = n // 4, (3 * n) // 4
    bottom_idx = mid_start + int(np.argmin(close[mid_start:mid_end]))
    bottom = close[bottom_idx]

    right_search_start = max(bottom_idx + 1, mid_end - 5)
    if right_search_start >= n:
        return ChartPattern(name="Cup and Handle", detected=False, confidence=0.0)
    right_idx = right_search_start + int(np.argmax(close[right_search_start:]))
    right_peak = close[right_idx]

    cup_depth = (left_peak - bottom) / left_peak if left_peak else 0
    rim_symmetry = 1 - abs(left_peak - right_peak) / left_peak if left_peak else 0
    recovered = right_peak >= left_peak * 0.95

    handle = close[right_idx:] if right_idx < n - 1 else np.array([])
    handle_pullback = None
    stage = "cup"
    handle_ok = False
    if len(handle) >= 3:
        handle_low = handle.min()
        handle_pullback = (right_peak - handle_low) / right_peak if right_peak else 0
        handle_ok = 0.03 <= handle_pullback <= 0.20
        stage = "handle" if handle_ok else "cup"

    detected = cup_depth >= 0.12 and recovered and rim_symmetry > 0.85 and (
        len(handle) < 3 or handle_ok
    )

    confidence = 0.0
    if detected:
        confidence = round(min(1.0, 0.4 + cup_depth * 0.5 + rim_symmetry * 0.2), 2)
        if handle_ok:
            stage = "handle"
            confidence = min(1.0, confidence + 0.1)

    breakout_level = float(max(left_peak, right_peak)) if detected else None
    last_vol = sub["volume"].tail(5).mean()
    avg_vol = sub["volume"].mean()
    volume_confirmation = bool(last_vol > avg_vol) if detected else None

    return ChartPattern(
        name="Cup and Handle",
        detected=bool(detected),
        confidence=confidence,
        stage=stage if detected else None,
        breakout_level=breakout_level,
        volume_confirmation=volume_confirmation,
    )


def double_top(df: pd.DataFrame, window: int = 60, tolerance: float = 0.02) -> ChartPattern:
    sub = df.tail(window)
    close = sub["close"].to_numpy()
    maxima, minima = _local_extrema(close, order=4)
    if len(maxima) < 2:
        return ChartPattern(name="Double Top", detected=False, confidence=0.0)

    p1, p2 = maxima[-2], maxima[-1]
    peak1, peak2 = close[p1], close[p2]
    diff = abs(peak1 - peak2) / max(peak1, peak2)

    between_min = close[p1:p2].min() if p2 > p1 else close[-1]
    neckline_break = close[-1] < between_min

    detected = diff <= tolerance and (p2 - p1) >= 5 and neckline_break
    confidence = round(max(0.0, 1 - diff / tolerance) * 0.8 + (0.2 if neckline_break else 0), 2) if detected else 0.0

    return ChartPattern(
        name="Double Top",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(between_min) if detected else None,
    )


def double_bottom(df: pd.DataFrame, window: int = 60, tolerance: float = 0.02) -> ChartPattern:
    sub = df.tail(window)
    close = sub["close"].to_numpy()
    maxima, minima = _local_extrema(close, order=4)
    if len(minima) < 2:
        return ChartPattern(name="Double Bottom", detected=False, confidence=0.0)

    t1, t2 = minima[-2], minima[-1]
    trough1, trough2 = close[t1], close[t2]
    diff = abs(trough1 - trough2) / max(trough1, trough2)

    between_max = close[t1:t2].max() if t2 > t1 else close[-1]
    neckline_break = close[-1] > between_max

    detected = diff <= tolerance and (t2 - t1) >= 5 and neckline_break
    confidence = round(max(0.0, 1 - diff / tolerance) * 0.8 + (0.2 if neckline_break else 0), 2) if detected else 0.0

    return ChartPattern(
        name="Double Bottom",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(between_max) if detected else None,
    )


def head_and_shoulders(df: pd.DataFrame, window: int = 80, tolerance: float = 0.03) -> ChartPattern:
    sub = df.tail(window)
    close = sub["close"].to_numpy()
    maxima, minima = _local_extrema(close, order=4)
    if len(maxima) < 3:
        return ChartPattern(name="Head and Shoulders", detected=False, confidence=0.0)

    l, h, r = maxima[-3], maxima[-2], maxima[-1]
    left_s, head, right_s = close[l], close[h], close[r]

    is_head_higher = head > left_s and head > right_s
    shoulders_close = abs(left_s - right_s) / max(left_s, right_s) <= tolerance

    neckline_candidates = close[l:r]
    neckline = neckline_candidates.min() if len(neckline_candidates) else close[-1]
    broke_neckline = close[-1] < neckline

    detected = is_head_higher and shoulders_close and broke_neckline
    confidence = round(0.5 + (0.3 if shoulders_close else 0) + (0.2 if broke_neckline else 0), 2) if detected else 0.0

    return ChartPattern(
        name="Head and Shoulders",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(neckline) if detected else None,
    )


def inverse_head_and_shoulders(df: pd.DataFrame, window: int = 80, tolerance: float = 0.03) -> ChartPattern:
    sub = df.tail(window)
    close = sub["close"].to_numpy()
    maxima, minima = _local_extrema(close, order=4)
    if len(minima) < 3:
        return ChartPattern(name="Inverse Head and Shoulders", detected=False, confidence=0.0)

    l, h, r = minima[-3], minima[-2], minima[-1]
    left_s, head, right_s = close[l], close[h], close[r]

    is_head_lower = head < left_s and head < right_s
    shoulders_close = abs(left_s - right_s) / max(left_s, right_s) <= tolerance

    neckline_candidates = close[l:r]
    neckline = neckline_candidates.max() if len(neckline_candidates) else close[-1]
    broke_neckline = close[-1] > neckline

    detected = is_head_lower and shoulders_close and broke_neckline
    confidence = round(0.5 + (0.3 if shoulders_close else 0) + (0.2 if broke_neckline else 0), 2) if detected else 0.0

    return ChartPattern(
        name="Inverse Head and Shoulders",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(neckline) if detected else None,
    )


def _trendline_slope(idx: np.ndarray, vals: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    slope, _ = np.polyfit(idx, vals, 1)
    return float(slope)


def triangle(df: pd.DataFrame, window: int = 50) -> ChartPattern:
    """
    Detects ascending / descending / symmetrical triangle by fitting
    trendlines through recent swing highs and swing lows and comparing
    their slopes.
    """
    sub = df.tail(window)
    close = sub["close"].to_numpy()
    maxima, minima = _local_extrema(close, order=3)

    if len(maxima) < 2 or len(minima) < 2:
        return ChartPattern(name="Triangle", detected=False, confidence=0.0)

    high_slope = _trendline_slope(maxima, close[maxima])
    low_slope = _trendline_slope(minima, close[minima])

    flat_thresh = np.std(close) * 0.01

    if abs(high_slope) < flat_thresh and low_slope > flat_thresh:
        name, detected = "Ascending Triangle", True
    elif high_slope < -flat_thresh and abs(low_slope) < flat_thresh:
        name, detected = "Descending Triangle", True
    elif high_slope < -flat_thresh and low_slope > flat_thresh:
        name, detected = "Symmetrical Triangle", True
    else:
        name, detected = "Triangle", False

    confidence = round(min(1.0, (abs(high_slope) + abs(low_slope)) / (flat_thresh * 4 + 1e-9)), 2) if detected else 0.0
    breakout_level = float(close[maxima[-1]]) if detected else None

    return ChartPattern(name=name, detected=detected, confidence=confidence, breakout_level=breakout_level)


def flag_or_pennant(df: pd.DataFrame, pole_window: int = 10, consolidation_window: int = 15) -> ChartPattern:
    """
    A sharp directional move (the 'pole') followed by a tight,
    low-volatility consolidation (the 'flag'/'pennant').
    """
    total = pole_window + consolidation_window
    sub = df.tail(total)
    if len(sub) < total:
        return ChartPattern(name="Flag/Pennant", detected=False, confidence=0.0)

    pole = sub["close"].iloc[:pole_window]
    consolidation = sub["close"].iloc[pole_window:]

    pole_move = (pole.iloc[-1] - pole.iloc[0]) / pole.iloc[0]
    consolidation_range = (consolidation.max() - consolidation.min()) / consolidation.mean()

    strong_pole = abs(pole_move) >= 0.08
    tight_consolidation = consolidation_range <= 0.05

    detected = strong_pole and tight_consolidation
    name = "Bull Flag" if pole_move > 0 else "Bear Flag"
    confidence = round(min(1.0, abs(pole_move) * 2 + (0.05 - consolidation_range) * 4), 2) if detected else 0.0
    confidence = max(0.0, confidence)

    return ChartPattern(
        name=name if detected else "Flag/Pennant",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(consolidation.max()) if detected and pole_move > 0 else (
            float(consolidation.min()) if detected else None
        ),
    )


def rectangle(df: pd.DataFrame, window: int = 40, tolerance: float = 0.03) -> ChartPattern:
    sub = df.tail(window)
    high, low = sub["high"], sub["low"]

    resistance = high.quantile(0.9)
    support = low.quantile(0.1)
    band = (resistance - support) / support if support else 0

    touches_top = (high >= resistance * (1 - tolerance)).sum()
    touches_bottom = (low <= support * (1 + tolerance)).sum()

    detected = band <= 0.10 and touches_top >= 2 and touches_bottom >= 2
    confidence = round(min(1.0, (touches_top + touches_bottom) / 10), 2) if detected else 0.0

    return ChartPattern(
        name="Rectangle",
        detected=bool(detected),
        confidence=confidence,
        breakout_level=float(resistance) if detected else None,
    )


DETECTORS = [
    cup_and_handle,
    double_top,
    double_bottom,
    head_and_shoulders,
    inverse_head_and_shoulders,
    triangle,
    flag_or_pennant,
    rectangle,
]


def detect_all(df: pd.DataFrame) -> List[ChartPattern]:
    if len(df) < 30:
        return []
    results = []
    for detector in DETECTORS:
        try:
            result = detector(df)
        except (IndexError, ValueError, ZeroDivisionError):
            continue
        if result.detected:
            results.append(result)
    return results

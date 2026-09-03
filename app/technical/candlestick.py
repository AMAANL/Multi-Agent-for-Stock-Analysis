"""
Deterministic candlestick pattern detection.

Every function takes the OHLCV DataFrame and looks only at the last
N candles it needs. No LLM calls here on purpose: the goal is that a
pattern is either mathematically true or not, and Grok's job later is
only to *explain* it, not decide whether it exists.
"""
from __future__ import annotations

from typing import List
import pandas as pd

from app.models.schemas import CandlestickPattern


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return max(row["high"] - row["low"], 1e-9)


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row) -> bool:
    return row["close"] < row["open"]


def doji(df: pd.DataFrame, threshold: float = 0.1) -> CandlestickPattern:
    c = df.iloc[-1]
    body_ratio = _body(c) / _range(c)
    detected = body_ratio <= threshold
    confidence = round(max(0.0, 1 - body_ratio / threshold), 2) if detected else 0.0
    return CandlestickPattern(name="Doji", detected=detected, confidence=confidence)


def hammer(df: pd.DataFrame) -> CandlestickPattern:
    c = df.iloc[-1]
    body = _body(c)
    lower_wick = min(c["open"], c["close"]) - c["low"]
    upper_wick = c["high"] - max(c["open"], c["close"])
    rng = _range(c)

    detected = (
        body / rng < 0.35
        and lower_wick >= 2 * body
        and upper_wick <= body * 0.5
    )
    confidence = round(min(1.0, lower_wick / rng), 2) if detected else 0.0
    return CandlestickPattern(name="Hammer", detected=detected, confidence=confidence)


def shooting_star(df: pd.DataFrame) -> CandlestickPattern:
    c = df.iloc[-1]
    body = _body(c)
    upper_wick = c["high"] - max(c["open"], c["close"])
    lower_wick = min(c["open"], c["close"]) - c["low"]
    rng = _range(c)

    detected = (
        body / rng < 0.35
        and upper_wick >= 2 * body
        and lower_wick <= body * 0.5
    )
    confidence = round(min(1.0, upper_wick / rng), 2) if detected else 0.0
    return CandlestickPattern(name="Shooting Star", detected=detected, confidence=confidence)


def bullish_engulfing(df: pd.DataFrame) -> CandlestickPattern:
    a, b = df.iloc[-2], df.iloc[-1]
    detected = (
        _is_bearish(a)
        and _is_bullish(b)
        and b["open"] <= a["close"]
        and b["close"] >= a["open"]
    )
    confidence = round(min(1.0, _body(b) / (_body(a) + 1e-9) / 2), 2) if detected else 0.0
    return CandlestickPattern(name="Bullish Engulfing", detected=detected, confidence=confidence)


def bearish_engulfing(df: pd.DataFrame) -> CandlestickPattern:
    a, b = df.iloc[-2], df.iloc[-1]
    detected = (
        _is_bullish(a)
        and _is_bearish(b)
        and b["open"] >= a["close"]
        and b["close"] <= a["open"]
    )
    confidence = round(min(1.0, _body(b) / (_body(a) + 1e-9) / 2), 2) if detected else 0.0
    return CandlestickPattern(name="Bearish Engulfing", detected=detected, confidence=confidence)


def morning_star(df: pd.DataFrame) -> CandlestickPattern:
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    first_bearish = _is_bearish(a)
    small_middle = _body(b) < _body(a) * 0.4
    third_bullish = _is_bullish(c)
    recovery = c["close"] > (a["open"] + a["close"]) / 2

    detected = first_bearish and small_middle and third_bullish and recovery
    confidence = round(min(1.0, (c["close"] - a["close"]) / _range(a)), 2) if detected else 0.0
    confidence = max(0.0, confidence)
    return CandlestickPattern(name="Morning Star", detected=detected, confidence=confidence)


def evening_star(df: pd.DataFrame) -> CandlestickPattern:
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    first_bullish = _is_bullish(a)
    small_middle = _body(b) < _body(a) * 0.4
    third_bearish = _is_bearish(c)
    reversal = c["close"] < (a["open"] + a["close"]) / 2

    detected = first_bullish and small_middle and third_bearish and reversal
    confidence = round(min(1.0, (a["close"] - c["close"]) / _range(a)), 2) if detected else 0.0
    confidence = max(0.0, confidence)
    return CandlestickPattern(name="Evening Star", detected=detected, confidence=confidence)


def three_white_soldiers(df: pd.DataFrame) -> CandlestickPattern:
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    detected = (
        _is_bullish(a) and _is_bullish(b) and _is_bullish(c)
        and b["open"] > a["open"] and b["close"] > a["close"]
        and c["open"] > b["open"] and c["close"] > b["close"]
        and all(_body(row) / _range(row) > 0.5 for row in (a, b, c))
    )
    confidence = 0.8 if detected else 0.0
    return CandlestickPattern(name="Three White Soldiers", detected=detected, confidence=confidence)


def three_black_crows(df: pd.DataFrame) -> CandlestickPattern:
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    detected = (
        _is_bearish(a) and _is_bearish(b) and _is_bearish(c)
        and b["open"] < a["open"] and b["close"] < a["close"]
        and c["open"] < b["open"] and c["close"] < b["close"]
        and all(_body(row) / _range(row) > 0.5 for row in (a, b, c))
    )
    confidence = 0.8 if detected else 0.0
    return CandlestickPattern(name="Three Black Crows", detected=detected, confidence=confidence)


DETECTORS = [
    doji,
    hammer,
    shooting_star,
    bullish_engulfing,
    bearish_engulfing,
    morning_star,
    evening_star,
    three_white_soldiers,
    three_black_crows,
]


def detect_all(df: pd.DataFrame) -> List[CandlestickPattern]:
    """Run every candlestick detector and return only the ones that fired
    (plus, always, at least an empty list if none did)."""
    if len(df) < 3:
        return []

    results = []
    for detector in DETECTORS:
        try:
            result = detector(df)
        except (IndexError, KeyError, ZeroDivisionError):
            continue
        if result.detected:
            results.append(result)
    return results

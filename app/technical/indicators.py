"""
Deterministic technical indicators (RSI, MACD, moving averages, ATR,
Bollinger Bands). Pure pandas/numpy math — no LLM involved, so results
are reproducible and unit-testable.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].rolling(window=period, min_periods=period).mean()


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].ewm(span=period, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # avg_loss == 0: all gains -> RSI 100 (unless avg_gain is also 0 -> flat -> 50)
    result = result.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    result = pd.Series(result, index=df.index)
    return result.fillna(50)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close"):
    ema_fast = ema(df, fast, column)
    ema_slow = ema(df, slow, column)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0, column: str = "close"):
    mid = sma(df, period, column)
    std = df[column].rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period, min_periods=period).mean()


def trend_direction(df: pd.DataFrame) -> str:
    """Simple trend read: price vs 50/200 SMA plus recent slope."""
    if len(df) < 60:
        return "neutral"

    sma50 = sma(df, 50).iloc[-1]
    sma200 = sma(df, min(200, len(df) - 1)).iloc[-1]
    close = df["close"].iloc[-1]

    if pd.isna(sma50) or pd.isna(sma200):
        return "neutral"

    if close > sma50 > sma200:
        return "bullish"
    if close < sma50 < sma200:
        return "bearish"
    return "neutral"

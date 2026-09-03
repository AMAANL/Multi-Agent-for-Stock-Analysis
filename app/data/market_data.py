"""
OHLCV price data access.

Wraps yfinance so the rest of the app depends on a small, stable
interface instead of the third-party library directly. Swapping this
for a paid data vendor later only means editing this one file.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

_TIMEFRAME_MAP = {
    "1d": ("1y", "1d"),
    "1wk": ("5y", "1wk"),
    "1mo": ("max", "1mo"),
}


def get_ohlcv(ticker: str, timeframe: str = "1d", period: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch OHLCV data for `ticker`.

    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume (lowercase, for consistency
    with the pattern-detection functions).
    """
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`.")

    default_period, interval = _TIMEFRAME_MAP.get(timeframe, ("1y", timeframe))
    period = period or default_period

    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for '{ticker}'. Check the ticker symbol.")

    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "date"
    return df


def get_latest_price(ticker: str) -> Optional[float]:
    df = get_ohlcv(ticker, timeframe="1d", period="5d")
    if df.empty:
        return None
    return float(df["close"].iloc[-1])

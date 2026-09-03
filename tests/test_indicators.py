import numpy as np
import pandas as pd
import pytest

from app.technical import indicators


def make_df(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    closes = pd.Series(closes, dtype=float)
    highs = pd.Series(highs, dtype=float) if highs is not None else closes + 1
    lows = pd.Series(lows, dtype=float) if lows is not None else closes - 1
    opens = closes.shift(1).fillna(closes.iloc[0])
    volumes = pd.Series(volumes, dtype=float) if volumes is not None else pd.Series([1000.0] * n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def test_sma_basic():
    df = make_df([1, 2, 3, 4, 5])
    result = indicators.sma(df, 3)
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    assert result.iloc[-1] == pytest.approx((3 + 4 + 5) / 3)


def test_rsi_all_gains_is_high():
    df = make_df(list(range(1, 40)))  # strictly increasing
    result = indicators.rsi(df, period=14)
    assert result.iloc[-1] > 90


def test_rsi_all_losses_is_low():
    df = make_df(list(range(40, 1, -1)))  # strictly decreasing
    result = indicators.rsi(df, period=14)
    assert result.iloc[-1] < 10


def test_macd_shapes():
    df = make_df(np.sin(np.linspace(0, 10, 100)) * 10 + 100)
    macd_line, signal_line, hist = indicators.macd(df)
    assert len(macd_line) == len(df)
    assert len(signal_line) == len(df)
    assert (hist == macd_line - signal_line).all()


def test_atr_nonnegative():
    df = make_df(np.random.default_rng(0).normal(100, 2, 60).cumsum())
    result = indicators.atr(df)
    assert (result.dropna() >= 0).all()


def test_trend_direction_bullish():
    closes = np.linspace(100, 200, 250)  # strong steady uptrend
    df = make_df(closes)
    assert indicators.trend_direction(df) == "bullish"


def test_trend_direction_bearish():
    closes = np.linspace(200, 100, 250)  # strong steady downtrend
    df = make_df(closes)
    assert indicators.trend_direction(df) == "bearish"


def test_trend_direction_short_series_is_neutral():
    df = make_df([100, 101, 102])
    assert indicators.trend_direction(df) == "neutral"

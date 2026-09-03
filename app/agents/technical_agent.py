"""
Technical / Chart Pattern Agent.

Runs every deterministic detector (indicators, candlesticks, chart
patterns, support/resistance) over the OHLCV data. No LLM call here —
Grok only gets to *narrate* the results in the Analyst Agent.
"""
from __future__ import annotations

import pandas as pd

from app.models.schemas import TechnicalResult
from app.technical import candlestick, chart_patterns, indicators, support_resistance


def run_technical_agent(df: pd.DataFrame) -> TechnicalResult:
    if len(df) < 30:
        raise ValueError("Not enough OHLCV data to run technical analysis (need 30+ candles).")

    rsi_series = indicators.rsi(df)
    macd_line, signal_line, _ = indicators.macd(df)
    sma50 = indicators.sma(df, 50)
    sma200 = indicators.sma(df, min(200, len(df) - 1))
    atr_series = indicators.atr(df)

    trend = indicators.trend_direction(df)

    candle_patterns = candlestick.detect_all(df)
    patterns = chart_patterns.detect_all(df)
    levels = support_resistance.find_support_resistance(df)

    latest_close = df["close"].iloc[-1]

    return TechnicalResult(
        trend=trend,
        rsi=round(float(rsi_series.iloc[-1]), 2) if not rsi_series.empty else None,
        macd=round(float(macd_line.iloc[-1]), 4) if not macd_line.empty else None,
        macd_signal=round(float(signal_line.iloc[-1]), 4) if not signal_line.empty else None,
        above_200_sma=bool(latest_close > sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None,
        above_50_sma=bool(latest_close > sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None,
        atr=round(float(atr_series.iloc[-1]), 2) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else None,
        candlestick_patterns=candle_patterns,
        chart_patterns=patterns,
        support_resistance=levels,
    )

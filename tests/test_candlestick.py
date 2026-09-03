import pandas as pd

from app.technical import candlestick


def row(open_, high, low, close):
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1000}


def test_morning_star_detected():
    rows = [
        row(100, 101, 90, 92),   # long bearish candle
        row(90, 92, 88, 91),      # small indecisive candle
        row(93, 100, 92, 99),     # strong bullish candle recovering above midpoint of candle 1
    ]
    df = pd.DataFrame(rows)
    result = candlestick.morning_star(df)
    assert result.detected is True
    assert result.name == "Morning Star"
    assert 0 <= result.confidence <= 1


def test_morning_star_not_detected_on_uptrend():
    rows = [
        row(90, 95, 89, 94),
        row(94, 98, 93, 97),
        row(97, 102, 96, 101),
    ]
    df = pd.DataFrame(rows)
    result = candlestick.morning_star(df)
    assert result.detected is False


def test_evening_star_detected():
    rows = [
        row(90, 100, 89, 98),    # long bullish candle
        row(98, 101, 97, 99),     # small indecisive candle
        row(97, 98, 88, 89),      # strong bearish candle, closes below midpoint of candle 1
    ]
    df = pd.DataFrame(rows)
    result = candlestick.evening_star(df)
    assert result.detected is True


def test_bullish_engulfing():
    rows = [
        row(50, 51, 45, 46),   # bearish
        row(45, 55, 44, 54),   # bullish, engulfs candle 1's body
    ]
    df = pd.DataFrame(rows)
    result = candlestick.bullish_engulfing(df)
    assert result.detected is True


def test_bearish_engulfing():
    rows = [
        row(45, 51, 44, 50),   # bullish
        row(51, 52, 40, 41),   # bearish, engulfs candle 1's body
    ]
    df = pd.DataFrame(rows)
    result = candlestick.bearish_engulfing(df)
    assert result.detected is True


def test_doji():
    rows = [row(100, 105, 95, 100.2)]
    df = pd.DataFrame(rows)
    result = candlestick.doji(df)
    assert result.detected is True


def test_hammer():
    rows = [row(100, 100.5, 85, 99)]
    df = pd.DataFrame(rows)
    result = candlestick.hammer(df)
    assert result.detected is True


def test_detect_all_returns_only_positive_matches():
    rows = [
        row(100, 101, 90, 92),
        row(90, 92, 88, 91),
        row(93, 100, 92, 99),
    ]
    df = pd.DataFrame(rows)
    results = candlestick.detect_all(df)
    assert all(r.detected for r in results)
    names = {r.name for r in results}
    assert "Morning Star" in names


def test_detect_all_short_dataframe_returns_empty():
    df = pd.DataFrame([row(100, 101, 99, 100)])
    assert candlestick.detect_all(df) == []

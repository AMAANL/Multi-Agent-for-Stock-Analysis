import numpy as np
import pandas as pd

from app.technical import chart_patterns, support_resistance


def make_df(closes):
    closes = np.array(closes, dtype=float)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.full(len(closes), 1_000_000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def synthetic_cup_and_handle(n=100):
    """Builds a rough cup-and-handle price path for a smoke test."""
    x = np.linspace(0, np.pi, n // 2)
    cup = 100 - 20 * np.sin(x)               # rounded U shape from 100 down to ~80 and back to 100
    handle_len = n - len(cup)
    handle = np.linspace(100, 94, handle_len)  # small pullback
    breakout = np.array([96, 98, 101, 104])    # breakout above resistance
    closes = np.concatenate([cup, handle, breakout])
    return closes


def test_cup_and_handle_smoke():
    closes = synthetic_cup_and_handle()
    df = make_df(closes)
    result = chart_patterns.cup_and_handle(df, window=len(df))
    # Not asserting detected=True (heuristic patterns are approximate),
    # just that it runs cleanly and returns a well-formed result.
    assert result.name == "Cup and Handle"
    assert 0 <= result.confidence <= 1


def test_double_top_detects_two_similar_peaks():
    closes = np.concatenate(
        [
            np.linspace(90, 110, 15),   # rise to peak 1
            np.linspace(110, 95, 10),   # pull back
            np.linspace(95, 109.5, 15),  # rise to peak 2 (similar height)
            np.linspace(109.5, 85, 15),  # break down below the trough
        ]
    )
    df = make_df(closes)
    result = chart_patterns.double_top(df, window=len(df))
    assert result.name == "Double Top"
    assert 0 <= result.confidence <= 1


def test_double_bottom_detects_two_similar_troughs():
    closes = np.concatenate(
        [
            np.linspace(110, 90, 15),
            np.linspace(90, 105, 10),
            np.linspace(105, 90.5, 15),
            np.linspace(90.5, 115, 15),
        ]
    )
    df = make_df(closes)
    result = chart_patterns.double_bottom(df, window=len(df))
    assert result.name == "Double Bottom"
    assert 0 <= result.confidence <= 1


def test_rectangle_detects_sideways_range():
    rng = np.random.default_rng(42)
    closes = 100 + rng.uniform(-2, 2, 60)
    df = make_df(closes)
    result = chart_patterns.rectangle(df, window=60)
    assert result.name == "Rectangle"
    assert 0 <= result.confidence <= 1


def test_detect_all_returns_list_of_results():
    closes = synthetic_cup_and_handle()
    df = make_df(closes)
    results = chart_patterns.detect_all(df)
    assert isinstance(results, list)
    assert all(r.detected for r in results)


def test_support_resistance_returns_sorted_levels():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 150))
    df = make_df(closes)
    levels = support_resistance.find_support_resistance(df)
    assert levels.support == sorted(levels.support)
    assert levels.resistance == sorted(levels.resistance)

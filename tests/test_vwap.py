"""
Tests for backend.indicators.vwap.

Compatible with pytest (run: pytest tests/test_vwap.py -v)
and also executable standalone: python tests/test_vwap.py

Manual VWAP reference values used in assertions
------------------------------------------------
Typical price  tp = (H + L + C) / 3

Session VWAP — 2-day dataset (bars 0-1 on day 1, bars 2-3 on day 2):
  bar 0: H=10, L=8,  C=9  → tp=9.0,  vol=100 → cum_tp_vol=900,  cum_vol=100 → vwap=9.0
  bar 1: H=11, L=9,  C=10 → tp=10.0, vol=200 → cum_tp_vol=2900, cum_vol=300 → vwap=9.6̄
  bar 2: H=12, L=10, C=11 → tp=11.0, vol=150 → cum_tp_vol=1650, cum_vol=150 → vwap=11.0  (reset)
  bar 3: H=13, L=11, C=12 → tp=12.0, vol=100 → cum_tp_vol=2850, cum_vol=250 → vwap=11.4

Anchored VWAP — same 4 bars, anchor at idx=1:
  idx=0 → NaN
  idx=1: tp=10.0, vol=200 → cum_tp_vol=2000, cum_vol=200 → avwap=10.0
  idx=2: tp=11.0, vol=150 → cum_tp_vol=3650, cum_vol=350 → avwap=3650/350≈10.4286
  idx=3: tp=12.0, vol=100 → cum_tp_vol=4850, cum_vol=450 → avwap=4850/450≈10.7778
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backend.indicators.vwap import (
    add_session_vwap,
    add_anchored_vwap,
    get_last_major_pivot_index,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_D1 = pd.Timestamp("2024-01-01")
_D2 = pd.Timestamp("2024-01-02")

# Intraday timestamps: two bars each on two consecutive days.
_TS_INTRADAY = [
    pd.Timestamp("2024-01-01 09:30"),
    pd.Timestamp("2024-01-01 10:00"),
    pd.Timestamp("2024-01-02 09:30"),
    pd.Timestamp("2024-01-02 10:00"),
]


def _candles_4bar(include_volume: bool = True):
    """4-bar OHLCV dataset spanning 2 sessions (2 bars each)."""
    data = {
        "ts":    _TS_INTRADAY,
        "open":  [9.0,  10.0, 11.0, 12.0],
        "high":  [10.0, 11.0, 12.0, 13.0],
        "low":   [8.0,   9.0, 10.0, 11.0],
        "close": [9.0,  10.0, 11.0, 12.0],
    }
    if include_volume:
        data["volume"] = [100.0, 200.0, 150.0, 100.0]
    return pd.DataFrame(data)


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Session VWAP — correctness
# ---------------------------------------------------------------------------


def test_session_vwap_resets_each_day():
    """VWAP resets at each calendar-day boundary; values match manual computation."""
    df = _candles_4bar()
    result = add_session_vwap(df)
    v = result["vwap_session"].to_list()

    assert _approx(v[0], 9.0),                 f"bar0: expected 9.0 got {v[0]}"
    assert _approx(v[1], 2900 / 300),           f"bar1: expected {2900/300} got {v[1]}"
    assert _approx(v[2], 11.0),                 f"bar2: expected 11.0 (reset) got {v[2]}"
    assert _approx(v[3], 2850 / 250),           f"bar3: expected {2850/250} got {v[3]}"


def test_session_vwap_single_bar_equals_typical_price():
    """Single bar per session: VWAP == (H+L+C)/3."""
    df = pd.DataFrame(
        {
            "ts":     [pd.Timestamp("2024-01-01 09:30")],
            "open":   [9.0],
            "high":   [10.0],
            "low":    [8.0],
            "close":  [9.0],
            "volume": [250.0],
        }
    )
    result = add_session_vwap(df)
    expected_tp = (10.0 + 8.0 + 9.0) / 3.0
    assert _approx(result["vwap_session"].iloc[0], expected_tp)


def test_session_vwap_missing_volume_all_nan():
    """If volume column is absent entirely, vwap_session is all-NaN."""
    df = _candles_4bar(include_volume=False)
    result = add_session_vwap(df)
    assert "vwap_session" in result.columns
    assert result["vwap_session"].isna().all()


def test_session_vwap_all_zero_volume_is_nan():
    """All-zero volume within a session: no positive volume → NaN."""
    df = _candles_4bar()
    df["volume"] = 0.0
    result = add_session_vwap(df)
    assert result["vwap_session"].isna().all()


def test_session_vwap_partial_zero_volume():
    """
    First bar of a session has zero volume (NaN VWAP); second bar has real
    volume and VWAP equals that bar's typical price (only source of weight).
    """
    df = pd.DataFrame(
        {
            "ts":     [
                pd.Timestamp("2024-01-01 09:30"),
                pd.Timestamp("2024-01-01 10:00"),
            ],
            "open":   [9.0, 10.0],
            "high":   [10.0, 12.0],
            "low":    [8.0, 10.0],
            "close":  [9.0, 11.0],
            "volume": [0.0, 100.0],
        }
    )
    result = add_session_vwap(df)
    v = result["vwap_session"].to_list()

    # bar 0: zero volume → NaN
    assert _is_nan(v[0]), f"bar0 should be NaN, got {v[0]}"
    # bar 1: only bar with volume → VWAP == tp of bar 1
    expected_tp1 = (12.0 + 10.0 + 11.0) / 3.0
    assert _approx(v[1], expected_tp1), f"bar1: expected {expected_tp1} got {v[1]}"


def test_session_vwap_does_not_mutate_input():
    """Input DataFrame is not modified."""
    df = _candles_4bar()
    cols_before = set(df.columns)
    add_session_vwap(df)
    assert set(df.columns) == cols_before


def test_session_vwap_weighted_by_volume():
    """
    Explicit weight check: second bar doubles first bar's volume; VWAP
    must be closer to bar-2 typical price.
    """
    df = pd.DataFrame(
        {
            "ts":     [
                pd.Timestamp("2024-01-01 09:30"),
                pd.Timestamp("2024-01-01 10:00"),
            ],
            "open":   [10.0, 20.0],
            "high":   [10.0, 20.0],
            "low":    [10.0, 20.0],
            "close":  [10.0, 20.0],
            "volume": [100.0, 200.0],
        }
    )
    # tp_bar0 = 10.0, tp_bar1 = 20.0
    # After bar 1: vwap = (10*100 + 20*200) / (100+200) = 5000/300 = 16.667
    result = add_session_vwap(df)
    expected = (10.0 * 100 + 20.0 * 200) / (100 + 200)
    assert _approx(result["vwap_session"].iloc[1], expected)


# ---------------------------------------------------------------------------
# Anchored VWAP — correctness
# ---------------------------------------------------------------------------


def test_anchored_vwap_anchor_at_start_covers_all():
    """Anchor at idx=0: all bars have vwap_anchored, matches cumulative VWAP."""
    df = _candles_4bar()
    result = add_anchored_vwap(df, anchor_idx=0)
    v = result["vwap_anchored"].to_list()

    # bar 0: avwap = tp0 = 9.0
    assert _approx(v[0], 9.0)
    # bar 1: avwap = (9*100 + 10*200) / (100+200) = 2900/300
    assert _approx(v[1], 2900 / 300)
    # bar 2: avwap = (2900 + 11*150) / (300+150) = 4550/450
    assert _approx(v[2], (2900 + 11 * 150) / 450)
    # bar 3: avwap = (4550 + 12*100) / (450+100) = 5750/550
    assert _approx(v[3], (4550 + 12 * 100) / 550)


def test_anchored_vwap_pre_anchor_rows_are_nan():
    """Rows before anchor_idx must be NaN."""
    df = _candles_4bar()
    result = add_anchored_vwap(df, anchor_idx=2)

    assert _is_nan(result["vwap_anchored"].iloc[0])
    assert _is_nan(result["vwap_anchored"].iloc[1])
    assert not _is_nan(result["vwap_anchored"].iloc[2])
    assert not _is_nan(result["vwap_anchored"].iloc[3])


def test_anchored_vwap_anchor_at_mid_values():
    """Anchored at idx=1: values match manual computation from the docstring."""
    df = _candles_4bar()
    result = add_anchored_vwap(df, anchor_idx=1)
    v = result["vwap_anchored"].to_list()

    assert _is_nan(v[0])
    assert _approx(v[1], 10.0)                   # 2000/200
    assert _approx(v[2], 3650 / 350)             # ≈ 10.4286
    assert _approx(v[3], 4850 / 450)             # ≈ 10.7778


def test_anchored_vwap_anchor_at_last_bar():
    """Anchor at last bar: only that bar has a value == typical price."""
    df = _candles_4bar()
    result = add_anchored_vwap(df, anchor_idx=3)
    v = result["vwap_anchored"].to_list()

    assert _is_nan(v[0])
    assert _is_nan(v[1])
    assert _is_nan(v[2])
    expected_tp = (13.0 + 11.0 + 12.0) / 3.0
    assert _approx(v[3], expected_tp)


def test_anchored_vwap_negative_anchor_idx():
    """Negative anchor_idx is resolved from the tail like Python indexing."""
    df = _candles_4bar()
    result_neg = add_anchored_vwap(df, anchor_idx=-2)
    result_pos = add_anchored_vwap(df, anchor_idx=2)
    # Both should produce identical output.
    pd.testing.assert_series_equal(
        result_neg["vwap_anchored"], result_pos["vwap_anchored"]
    )


def test_anchored_vwap_missing_volume_all_nan():
    """If volume column is absent, vwap_anchored is all-NaN."""
    df = _candles_4bar(include_volume=False)
    result = add_anchored_vwap(df, anchor_idx=0)
    assert result["vwap_anchored"].isna().all()


def test_anchored_vwap_zero_volume_anchor_bar():
    """
    When the anchor bar itself has zero volume, its AVWAP is NaN; the next
    positive-volume bar starts the cumulation correctly.
    """
    df = _candles_4bar()
    df.loc[df.index[1], "volume"] = 0.0  # zero out bar at idx=1

    # Anchor at idx=1 (zero volume).  idx=2 and idx=3 still have volume.
    result = add_anchored_vwap(df, anchor_idx=1)
    v = result["vwap_anchored"].to_list()

    assert _is_nan(v[0])    # before anchor
    assert _is_nan(v[1])    # anchor bar: zero volume → no valid weight yet
    # bar 2: vol=150, tp=11 → only bar with weight so far → avwap = tp2
    expected_tp2 = (12.0 + 10.0 + 11.0) / 3.0
    assert _approx(v[2], expected_tp2), f"bar2 avwap: expected {expected_tp2} got {v[2]}"


def test_anchored_vwap_out_of_range_raises():
    """anchor_idx beyond the DataFrame length raises ValueError."""
    df = _candles_4bar()
    try:
        add_anchored_vwap(df, anchor_idx=10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of range" in str(e)


def test_anchored_vwap_does_not_mutate_input():
    df = _candles_4bar()
    cols_before = set(df.columns)
    add_anchored_vwap(df, anchor_idx=0)
    assert set(df.columns) == cols_before


# ---------------------------------------------------------------------------
# get_last_major_pivot_index
# ---------------------------------------------------------------------------


def _df_with_pivots(pivot_lows=None, pivot_highs=None, n=30):
    """
    Build a minimal DataFrame with pivot_low / pivot_high columns.

    pivot_lows / pivot_highs: list of (iloc_position, low_price, high_price, close_price)
    All other bars have close=100, high=101, low=99.
    """
    highs  = np.full(n, 101.0)
    lows   = np.full(n, 99.0)
    closes = np.full(n, 100.0)

    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    if pivot_lows:
        for pos, lo, hi, cl in pivot_lows:
            lows[pos]   = lo
            highs[pos]  = hi
            closes[pos] = cl
            pl[pos] = True

    if pivot_highs:
        for pos, lo, hi, cl in pivot_highs:
            lows[pos]   = lo
            highs[pos]  = hi
            closes[pos] = cl
            ph[pos] = True

    return pd.DataFrame(
        {
            "ts":         pd.date_range("2024-01-01", periods=n, freq="D"),
            "open":       closes,
            "high":       highs,
            "low":        lows,
            "close":      closes,
            "volume":     np.ones(n) * 100,
            "pivot_low":  pl,
            "pivot_high": ph,
        }
    )


def test_pivot_index_no_pivots_returns_none():
    """No pivot_low entries → None."""
    df = _df_with_pivots()  # no pivots set
    result = get_last_major_pivot_index(df, kind="low")
    assert result is None


def test_pivot_index_single_pivot_returns_it():
    """Only one pivot candidate → that position is returned regardless of prominence."""
    df = _df_with_pivots(pivot_lows=[(5, 90.0, 101.0, 95.0)])
    result = get_last_major_pivot_index(df, kind="low")
    assert result == 5


def test_pivot_index_prominence_selects_larger_swing():
    """
    Two pivot_lows; the one that caused the larger upswing wins.

    pivot_A at idx=5:  low=90, followed by close=100 → swing = 100-90 = 10
    pivot_B at idx=15: low=98, followed by close=100 → swing = 100-98 = 2

    Expected: pivot_A at idx=5 wins (larger swing magnitude).
    """
    df = _df_with_pivots(
        pivot_lows=[
            (5,  90.0, 101.0, 90.0),   # deep low → large swing
            (15, 98.0, 101.0, 98.0),   # shallow low → small swing
        ]
    )
    result = get_last_major_pivot_index(
        df, kind="low", lookback_pivots=5, prominence_horizon=10
    )
    assert result == 5, f"Expected idx=5 (deeper swing), got {result}"


def test_pivot_index_kind_high():
    """
    Two pivot_highs; the one with the larger downswing wins.

    pivot_A at idx=5:  high=110, followed by close=100 → swing = 110-100 = 10
    pivot_B at idx=15: high=102, followed by close=100 → swing = 102-100 = 2

    Expected: pivot_A at idx=5 wins.
    """
    df = _df_with_pivots(
        pivot_highs=[
            (5,  99.0, 110.0, 110.0),
            (15, 99.0, 102.0, 102.0),
        ]
    )
    result = get_last_major_pivot_index(
        df, kind="high", lookback_pivots=5, prominence_horizon=10
    )
    assert result == 5, f"Expected idx=5, got {result}"


def test_pivot_index_lookback_limits_candidates():
    """
    lookback_pivots=1 means only the most recent pivot is a candidate,
    even if an older pivot has larger prominence.
    """
    df = _df_with_pivots(
        pivot_lows=[
            (5,  80.0, 101.0, 80.0),   # old, very deep → prominence=20
            (25, 98.0, 101.0, 98.0),   # recent, shallow → prominence=2
        ]
    )
    # With lookback_pivots=1, only idx=25 is considered.
    result = get_last_major_pivot_index(
        df, kind="low", lookback_pivots=1, prominence_horizon=5
    )
    assert result == 25, f"Expected idx=25 (only candidate), got {result}"


def test_pivot_index_missing_column_raises():
    """Missing pivot_low column raises ValueError with a helpful message."""
    df = pd.DataFrame(
        {
            "ts":    pd.date_range("2024-01-01", periods=5, freq="D"),
            "high":  [101.0] * 5,
            "low":   [99.0] * 5,
            "close": [100.0] * 5,
        }
    )
    try:
        get_last_major_pivot_index(df, kind="low")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "pivot_low" in str(e)
        assert "find_pivots" in str(e)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_session_vwap_raises_on_missing_price_col():
    df = pd.DataFrame({"ts": [_D1], "high": [10.0], "low": [8.0]})  # no 'close'
    try:
        add_session_vwap(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "close" in str(e)


def test_session_vwap_raises_on_empty_df():
    df = pd.DataFrame(columns=["ts", "high", "low", "close", "volume"])
    try:
        add_session_vwap(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_anchored_vwap_raises_on_missing_price_col():
    df = pd.DataFrame({"ts": [_D1], "high": [10.0], "close": [9.0]})  # no 'low'
    try:
        add_anchored_vwap(df, anchor_idx=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "low" in str(e)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Session VWAP
        test_session_vwap_resets_each_day,
        test_session_vwap_single_bar_equals_typical_price,
        test_session_vwap_missing_volume_all_nan,
        test_session_vwap_all_zero_volume_is_nan,
        test_session_vwap_partial_zero_volume,
        test_session_vwap_does_not_mutate_input,
        test_session_vwap_weighted_by_volume,
        # Anchored VWAP
        test_anchored_vwap_anchor_at_start_covers_all,
        test_anchored_vwap_pre_anchor_rows_are_nan,
        test_anchored_vwap_anchor_at_mid_values,
        test_anchored_vwap_anchor_at_last_bar,
        test_anchored_vwap_negative_anchor_idx,
        test_anchored_vwap_missing_volume_all_nan,
        test_anchored_vwap_zero_volume_anchor_bar,
        test_anchored_vwap_out_of_range_raises,
        test_anchored_vwap_does_not_mutate_input,
        # get_last_major_pivot_index
        test_pivot_index_no_pivots_returns_none,
        test_pivot_index_single_pivot_returns_it,
        test_pivot_index_prominence_selects_larger_swing,
        test_pivot_index_kind_high,
        test_pivot_index_lookback_limits_candidates,
        test_pivot_index_missing_column_raises,
        # Validation
        test_session_vwap_raises_on_missing_price_col,
        test_session_vwap_raises_on_empty_df,
        test_anchored_vwap_raises_on_missing_price_col,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed:
        sys.exit(1)

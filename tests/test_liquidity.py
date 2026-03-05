"""
Tests for backend.signals.liquidity.detect_liquidity_sweeps.

Compatible with pytest (run: pytest tests/test_liquidity.py -v)
and also executable standalone: python tests/test_liquidity.py
"""

from __future__ import annotations

import sys
import os

# Allow running directly from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from backend.indicators.zones import Zone
from backend.signals.liquidity import detect_liquidity_sweeps, _nearest_zones


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T0 = pd.Timestamp("2024-01-01")
_T1 = pd.Timestamp("2024-01-02")
_T2 = pd.Timestamp("2024-01-03")


def _zone(kind, low, high, touches=3):
    """Construct a minimal Zone for testing."""
    return Zone(
        kind=kind,
        low=float(low),
        high=float(high),
        center=(low + high) / 2.0,
        touches=touches,
        first_ts=_T0,
        last_ts=_T1,
    )


def _candles(*rows):
    """
    Build a DataFrame from (ts, open, high, low, close) tuples.
    volume is omitted intentionally to verify volume-independence.
    """
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])


def _resistance(low=99.0, high=100.0, touches=3):
    return _zone("resistance", low, high, touches)


def _support(low=99.0, high=100.0, touches=3):
    return _zone("support", low, high, touches)


# ---------------------------------------------------------------------------
# Core sweep detection
# ---------------------------------------------------------------------------


def test_sweep_up_single_bar():
    """Wick above resistance AND close below zone.high on the same bar."""
    # resistance zone: low=99, high=100, zone_tol=0.5
    # pierce threshold = 100.5
    # bar: high=101 (pierces), close=98.5 (reclaims below zone.high=100)
    df = _candles(
        (_T0, 97, 98, 96, 97),
        (_T1, 98, 101.0, 97.5, 98.5),  # single-bar sweep
    )
    zones = [_resistance(low=99, high=100)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)

    assert result is not None
    assert result["type"] == "sweep_up"
    assert result["zone_kind"] == "resistance"
    assert result["zone_center"] == 99.5
    assert result["touches"] == 3
    assert result["close"] == 98.5
    assert result["ts"] == _T1


def test_sweep_up_two_bar():
    """Wick pierce on bar N-1; confirming close on bar N (last)."""
    # bar N-1: high=101.5 → pierces zone.high(100) + tol(0.5) = 100.5 ✓
    # bar N  : close=98 → reclaims below zone.high=100 ✓
    df = _candles(
        (_T0, 95, 96, 94, 95),
        (_T1, 98, 101.5, 97, 99.5),   # wick bar
        (_T2, 99, 100.2, 97, 98.0),   # confirming close bar
    )
    zones = [_resistance(low=99, high=100)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=2)

    assert result is not None
    assert result["type"] == "sweep_up"
    assert result["ts"] == _T2
    assert result["close"] == 98.0


def test_sweep_down_single_bar():
    """Wick below support AND close above zone.low on the same bar."""
    # support zone: low=100, high=101
    # pierce threshold = 100 - 0.5 = 99.5
    # bar: low=99.0 (pierces), close=101.5 (reclaims above zone.low=100)
    df = _candles(
        (_T0, 103, 104, 102, 103),
        (_T1, 102, 103, 99.0, 101.5),  # single-bar sweep
    )
    zones = [_support(low=100, high=101)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)

    assert result is not None
    assert result["type"] == "sweep_down"
    assert result["zone_kind"] == "support"
    assert result["close"] == 101.5


def test_sweep_down_two_bar():
    """Wick pierce on bar N-1; confirming close on bar N."""
    df = _candles(
        (_T0, 105, 106, 104, 105),
        (_T1, 103, 104, 98.5, 100.5),  # wick bar: low=98.5 < 99.5
        (_T2, 101, 103, 100.1, 102.0), # confirming close bar
    )
    zones = [_support(low=100, high=101)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=2)

    assert result is not None
    assert result["type"] == "sweep_down"
    assert result["ts"] == _T2


def test_no_sweep_when_close_does_not_reclaim_resistance():
    """Wick pierces resistance but close stays ABOVE zone.high → real breakout, not sweep."""
    # bar: high=101.5 (pierces 100+0.5=100.5), close=100.5 (still above zone.high=100)
    df = _candles(
        (_T0, 97, 98, 96, 97),
        (_T1, 98, 101.5, 98, 100.5),
    )
    zones = [_resistance(low=99, high=100)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)
    assert result is None


def test_no_sweep_when_close_does_not_reclaim_support():
    """Wick pierces support but close stays BELOW zone.low → real breakdown, not sweep."""
    df = _candles(
        (_T0, 105, 106, 104, 105),
        (_T1, 103, 103.5, 98.5, 99.4),  # low=98.5 pierces 99.5, close=99.4 < zone.low=100
    )
    zones = [_support(low=100, high=101)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)
    assert result is None


def test_no_sweep_when_wick_does_not_pierce_zone():
    """Wick touches zone boundary but doesn't exceed zone.high + tol."""
    # resistance zone high=100, tol=0.5 → pierce threshold=100.5
    # bar high=100.4 < 100.5 → no pierce
    df = _candles(
        (_T0, 97, 98, 96, 97),
        (_T1, 98, 100.4, 97, 98.0),
    )
    zones = [_resistance(low=99, high=100)]
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)
    assert result is None


def test_empty_zones_returns_none():
    df = _candles(
        (_T0, 97, 101, 96, 98),
    )
    result = detect_liquidity_sweeps(df, zones=[], zone_tol=0.5, lookback_bars=1)
    assert result is None


def test_volume_column_not_required():
    """Function must not crash when volume is absent."""
    df = _candles(
        (_T0, 97, 98, 96, 97),
        (_T1, 98, 102.0, 97, 98.5),
    )
    zones = [_resistance(low=99, high=100)]
    # Should not raise even though 'volume' is missing.
    result = detect_liquidity_sweeps(df, zones, zone_tol=0.5, lookback_bars=1)
    assert result is not None
    assert result["type"] == "sweep_up"


# ---------------------------------------------------------------------------
# Priority — deeper sweep wins
# ---------------------------------------------------------------------------


def test_priority_deeper_sweep_wins():
    """When two zones both produce sweep events, the deeper wick wins."""
    # current price after bar: 97
    # Zone A (resistance): high=100 → pierce_threshold=100.5, depth = max_high - 100
    # Zone B (resistance): high=98  → pierce_threshold=98.5,  depth = max_high - 98
    # bar high=102 → depth_A = 2.0, depth_B = 4.0 → Zone B wins
    df = _candles(
        (_T0, 95, 96, 94, 95),
        (_T1, 96, 102.0, 95, 97.0),
    )
    zone_a = _resistance(low=99, high=100, touches=5)
    zone_b = _resistance(low=97, high=98, touches=2)  # closer to price, deeper sweep
    result = detect_liquidity_sweeps(df, [zone_a, zone_b], zone_tol=0.5, lookback_bars=1)

    assert result is not None
    assert result["zone_center"] == 97.5  # zone_b center
    assert result["touches"] == 2


def test_priority_tiebreak_by_touches():
    """Equal depth: zone with more touches wins."""
    # Two resistance zones at same distance from max_high.
    # high=102 → depth for both zones is 1.0 (zone.high = 101 for both,
    # but we need different centers, so use zone.high = 101 and 101 but
    # different lows).  Actually let's use same high=101 for both.
    df = _candles(
        (_T0, 98, 99, 97, 98),
        (_T1, 99, 102.5, 98, 97.5),
    )
    zone_weak = _resistance(low=100, high=101, touches=2)    # depth = 102.5-101 = 1.5
    zone_strong = _resistance(low=100.5, high=101.5, touches=5)  # depth = 102.5-101.5 = 1.0

    # zone_weak has deeper sweep (1.5 > 1.0) → should win on depth
    result = detect_liquidity_sweeps(
        df, [zone_weak, zone_strong], zone_tol=0.5, lookback_bars=1
    )
    assert result is not None
    assert result["touches"] == 2  # zone_weak won by depth

    # Now make depths equal: zone_strong.high = 101, depth = 1.5 for both.
    zone_strong2 = _resistance(low=100, high=101, touches=5)
    result2 = detect_liquidity_sweeps(
        df, [zone_weak, zone_strong2], zone_tol=0.5, lookback_bars=1
    )
    assert result2 is not None
    assert result2["touches"] == 5  # tiebreak: more touches wins


# ---------------------------------------------------------------------------
# Proximity filter
# ---------------------------------------------------------------------------


def test_only_nearest_6_zones_checked():
    """A sweep-qualifying zone that falls outside the 6 nearest is ignored.

    Setup:
    - current close = 100.0
    - 6 nearby *support* zones centered at 91.5–96.5 (distance 3.5–8.5 from price)
    - 1 far *resistance* zone centered at 200 (distance 100 from price) — excluded
    - bar wick reaches 215 → would trigger sweep_up on the far resistance zone IF checked,
      but the far zone is the 7th (farthest) and is never evaluated.
    - bar low = 97.5 → does not pierce any support zone's lower boundary.
    Result must be None.
    """
    df = _candles(
        (_T0, 99, 99.5, 98.5, 100.0),
        (_T1, 100, 215.0, 97.5, 100.0),   # high=215 pierces far zone; low=97.5 safe
    )
    # Support zones near price (91.5–96.5); min_low=97.5 > zone.low-tol for all.
    nearby_zones = [_support(low=91 + i, high=92 + i, touches=2) for i in range(6)]
    # Far resistance zone: pierce_threshold = 200.5+0.5 = 201 < 215, would fire if checked.
    far_zone = _resistance(low=199.5, high=200.5, touches=5)

    result = detect_liquidity_sweeps(
        df, nearby_zones + [far_zone], zone_tol=0.5, lookback_bars=1
    )
    assert result is None


def test_nearest_zones_helper_returns_all_when_few():
    """_nearest_zones returns all zones when count <= n."""
    zones = [_resistance(low=i, high=i + 1) for i in range(4)]
    result = _nearest_zones(zones, price=5.0, n=6)
    assert result == zones  # same list, no filtering needed


def test_nearest_zones_helper_selects_closest():
    """_nearest_zones picks the n zones closest to price."""
    zones = [
        _zone("resistance", low=10, high=11),   # center 10.5
        _zone("resistance", low=20, high=21),   # center 20.5
        _zone("resistance", low=30, high=31),   # center 30.5
        _zone("resistance", low=40, high=41),   # center 40.5
    ]
    # price = 12 → distances: 1.5, 8.5, 18.5, 28.5 → closest 2: zones[0], zones[1]
    nearest = _nearest_zones(zones, price=12.0, n=2)
    centers = {z.center for z in nearest}
    assert 10.5 in centers
    assert 20.5 in centers
    assert len(nearest) == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_missing_columns():
    df = pd.DataFrame({"ts": [_T0], "high": [101.0], "low": [99.0]})  # missing 'close'
    try:
        detect_liquidity_sweeps(df, [], zone_tol=0.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "close" in str(e)


def test_raises_on_empty_dataframe():
    df = pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    try:
        detect_liquidity_sweeps(df, [], zone_tol=0.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_raises_on_bad_zone_tol():
    df = _candles((_T0, 97, 101, 96, 98))
    try:
        detect_liquidity_sweeps(df, [], zone_tol=0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zone_tol" in str(e)


def test_raises_on_bad_lookback():
    df = _candles((_T0, 97, 101, 96, 98))
    try:
        detect_liquidity_sweeps(df, [], zone_tol=0.5, lookback_bars=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_raises_when_df_smaller_than_lookback():
    df = _candles((_T0, 97, 101, 96, 98))  # 1 row
    try:
        detect_liquidity_sweeps(df, [], zone_tol=0.5, lookback_bars=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "lookback_bars" in str(e)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_sweep_up_single_bar,
        test_sweep_up_two_bar,
        test_sweep_down_single_bar,
        test_sweep_down_two_bar,
        test_no_sweep_when_close_does_not_reclaim_resistance,
        test_no_sweep_when_close_does_not_reclaim_support,
        test_no_sweep_when_wick_does_not_pierce_zone,
        test_empty_zones_returns_none,
        test_volume_column_not_required,
        test_priority_deeper_sweep_wins,
        test_priority_tiebreak_by_touches,
        test_only_nearest_6_zones_checked,
        test_nearest_zones_helper_returns_all_when_few,
        test_nearest_zones_helper_selects_closest,
        test_raises_on_missing_columns,
        test_raises_on_empty_dataframe,
        test_raises_on_bad_zone_tol,
        test_raises_on_bad_lookback,
        test_raises_when_df_smaller_than_lookback,
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

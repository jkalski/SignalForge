"""
Tests for backend.indicators.mtf.

Compatible with pytest (run: pytest tests/test_mtf.py -v)
and also executable standalone: python tests/test_mtf.py

Coverage
--------
align_zones        — linkage logic, many-to-one, htf_only, distance values
get_htf_bias       — bullish / bearish / neutral classifications
filter_signal_by_htf — all five decision rules including the sweep override
_near_any_htf_zone — proximity helper (indirectly via filter tests)
Validation         — tol <= 0 raises ValueError
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backend.indicators.zones import Zone
from backend.indicators.mtf import (
    align_zones,
    get_htf_bias,
    filter_signal_by_htf,
    _near_any_htf_zone,
)


# ---------------------------------------------------------------------------
# Shared Zone helpers
# ---------------------------------------------------------------------------


def _zone(kind, low, high, touches=3):
    return Zone(
        kind=kind,
        low=float(low),
        high=float(high),
        center=(low + high) / 2.0,
        touches=touches,
        first_ts=pd.Timestamp("2024-01-01"),
        last_ts=pd.Timestamp("2024-01-15"),
    )


def _res(low, high, touches=3):
    return _zone("resistance", low, high, touches)


def _sup(low, high, touches=3):
    return _zone("support", low, high, touches)


# ---------------------------------------------------------------------------
# align_zones — basic linkage
# ---------------------------------------------------------------------------


def test_align_linked_when_within_tol():
    """One LTF zone, one HTF zone, centers within tol → single linked entry."""
    htf = [_res(99.0, 101.0)]   # center = 100.0
    ltf = [_res(99.5, 101.5)]   # center = 100.5  → distance = 0.5

    result = align_zones(htf, ltf, tol=1.0)

    assert len(result["linked"]) == 1
    assert result["unlinked_ltf"] == []
    assert result["htf_only"] == []

    entry = result["linked"][0]
    assert entry["ltf"] is ltf[0]
    assert entry["htf"] is htf[0]
    assert abs(entry["distance"] - 0.5) < 1e-9


def test_align_unlinked_when_outside_tol():
    """Centers farther apart than tol → LTF goes to unlinked_ltf, HTF to htf_only."""
    htf = [_res(99.0, 101.0)]   # center = 100.0
    ltf = [_res(104.5, 105.5)]  # center = 105.0  → distance = 5.0

    result = align_zones(htf, ltf, tol=2.0)

    assert result["linked"] == []
    assert len(result["unlinked_ltf"]) == 1
    assert result["unlinked_ltf"][0] is ltf[0]
    assert len(result["htf_only"]) == 1
    assert result["htf_only"][0] is htf[0]


def test_align_boundary_exactly_at_tol():
    """Distance exactly equal to tol → treated as linked (≤, not <)."""
    htf = [_res(99.0, 101.0)]   # center = 100.0
    ltf = [_res(101.0, 103.0)]  # center = 102.0  → distance = 2.0

    result = align_zones(htf, ltf, tol=2.0)

    assert len(result["linked"]) == 1
    assert abs(result["linked"][0]["distance"] - 2.0) < 1e-9


def test_align_boundary_just_outside_tol():
    """Distance just beyond tol → unlinked."""
    htf = [_res(99.0, 101.0)]   # center = 100.0
    ltf = [_res(101.1, 103.1)]  # center = 102.1  → distance ≈ 2.1

    result = align_zones(htf, ltf, tol=2.0)

    assert result["linked"] == []
    assert len(result["unlinked_ltf"]) == 1


# ---------------------------------------------------------------------------
# align_zones — many-to-one and htf_only
# ---------------------------------------------------------------------------


def test_align_two_ltf_link_to_same_htf():
    """Two LTF zones close to the same HTF zone → two linked entries, htf_only empty."""
    htf = [_res(99.0, 101.0)]    # center = 100.0
    ltf = [
        _res(99.5, 101.5),       # center = 100.5, dist = 0.5
        _res(98.0, 100.0),       # center = 99.0,  dist = 1.0
    ]

    result = align_zones(htf, ltf, tol=2.0)

    assert len(result["linked"]) == 2
    assert result["unlinked_ltf"] == []
    assert result["htf_only"] == []
    # Both entries point to the same HTF zone.
    assert all(e["htf"] is htf[0] for e in result["linked"])


def test_align_htf_only_when_unmatched_htf():
    """HTF zones with no nearby LTF zone appear in htf_only."""
    htf = [
        _res(99.0, 101.0),       # center = 100.0  ← matched
        _res(199.0, 201.0),      # center = 200.0  ← far, unmatched
    ]
    ltf = [_res(99.5, 101.5)]    # center = 100.5 → links to htf[0]

    result = align_zones(htf, ltf, tol=2.0)

    assert len(result["linked"]) == 1
    assert result["linked"][0]["htf"] is htf[0]
    assert len(result["htf_only"]) == 1
    assert result["htf_only"][0] is htf[1]


def test_align_multiple_ltf_nearest_htf_selected():
    """Each LTF zone independently picks the nearest HTF zone."""
    htf = [
        _res(99.0,  101.0),      # center = 100.0
        _res(149.0, 151.0),      # center = 150.0
    ]
    ltf = [
        _res(99.5,  101.5),      # center = 100.5 → nearest htf[0] (dist 0.5)
        _res(149.5, 151.5),      # center = 150.5 → nearest htf[1] (dist 0.5)
    ]

    result = align_zones(htf, ltf, tol=5.0)

    assert len(result["linked"]) == 2
    centers_in_linked = {(e["ltf"].center, e["htf"].center) for e in result["linked"]}
    assert (100.5, 100.0) in centers_in_linked
    assert (150.5, 150.0) in centers_in_linked
    assert result["htf_only"] == []


def test_align_linked_sorted_by_distance():
    """linked list is sorted ascending by distance."""
    htf = [_res(99.0, 101.0)]   # center = 100
    ltf = [
        _res(98.0, 100.0),      # center = 99.0   dist = 1.0
        _res(99.5, 100.5),      # center = 100.0  dist = 0.0
        _res(101.0, 103.0),     # center = 102.0  dist = 2.0
    ]
    result = align_zones(htf, ltf, tol=3.0)

    dists = [e["distance"] for e in result["linked"]]
    assert dists == sorted(dists), f"Expected sorted distances, got {dists}"


# ---------------------------------------------------------------------------
# align_zones — edge cases (empty inputs)
# ---------------------------------------------------------------------------


def test_align_empty_htf_all_ltf_unlinked():
    ltf = [_res(99, 101), _sup(79, 81)]
    result = align_zones([], ltf, tol=2.0)

    assert result["linked"] == []
    assert result["htf_only"] == []
    assert len(result["unlinked_ltf"]) == 2


def test_align_empty_ltf_all_htf_only():
    htf = [_res(99, 101), _sup(79, 81)]
    result = align_zones(htf, [], tol=2.0)

    assert result["linked"] == []
    assert result["unlinked_ltf"] == []
    assert len(result["htf_only"]) == 2


def test_align_both_empty():
    result = align_zones([], [], tol=1.0)
    assert result == {"linked": [], "unlinked_ltf": [], "htf_only": []}


# ---------------------------------------------------------------------------
# align_zones — validation
# ---------------------------------------------------------------------------


def test_align_raises_on_nonpositive_tol():
    try:
        align_zones([], [], tol=0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "tol" in str(e)

    try:
        align_zones([], [], tol=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "tol" in str(e)


# ---------------------------------------------------------------------------
# get_htf_bias
# ---------------------------------------------------------------------------


def _htf_df(closes, n=60):
    """
    Build a minimal HTF DataFrame with a flat history + final close.

    The first (n-1) bars are all at 100.0 so the EMA50 converges to ≈100.
    The last bar uses the supplied close to push the bias.
    """
    all_closes = [100.0] * (n - 1) + [closes]
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "ts":     ts,
            "open":   all_closes,
            "high":   [c + 1.0 for c in all_closes],
            "low":    [c - 1.0 for c in all_closes],
            "close":  all_closes,
            "volume": [1000.0] * n,
        }
    )


def test_get_htf_bias_bullish():
    """close clearly above EMA50 → 'bullish'."""
    df = _htf_df(closes=130.0)
    assert get_htf_bias(df) == "bullish"


def test_get_htf_bias_bearish():
    """close clearly below EMA50 → 'bearish'."""
    df = _htf_df(closes=70.0)
    assert get_htf_bias(df) == "bearish"


def test_get_htf_bias_uses_only_last_bar():
    """
    Even if the penultimate bar is below EMA50, a last-bar close above it
    returns 'bullish'.  Bias is evaluated only on the most recent bar.
    """
    # 59 bars at 70 (→ EMA50 ≈ 70), then one big jump to 200.
    closes = [70.0] * 59 + [200.0]
    ts = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "ts":     ts,
            "open":   closes,
            "high":   [c + 1 for c in closes],
            "low":    [c - 1 for c in closes],
            "close":  closes,
            "volume": [1000.0] * 60,
        }
    )
    assert get_htf_bias(df) == "bullish"


# ---------------------------------------------------------------------------
# filter_signal_by_htf — basic rules
# ---------------------------------------------------------------------------


def _empty_alignment():
    return {"linked": [], "unlinked_ltf": [], "htf_only": []}


def _alignment_with_htf_at(center):
    """Alignment where one HTF zone is at the given center (htf_only, no LTF echo)."""
    z = _res(center - 0.5, center + 0.5)
    return {"linked": [], "unlinked_ltf": [], "htf_only": [z]}


def _signal(sig_type, zone_center=100.0):
    return {"type": sig_type, "zone_center": zone_center, "touches": 3}


def test_filter_none_signal_returns_false():
    result = filter_signal_by_htf(None, "bullish", _empty_alignment(), zone_tol=1.0)
    assert result is False


def test_filter_neutral_bias_allows_any_signal():
    for sig_type in [
        "breakout_up", "breakdown_down", "bounce_up",
        "reject_down", "sweep_up", "sweep_down",
    ]:
        result = filter_signal_by_htf(
            _signal(sig_type), "neutral", _empty_alignment(), zone_tol=1.0
        )
        assert result is True, f"Expected True for {sig_type} with neutral bias"


def test_filter_bullish_bias_allows_bullish_signals():
    for sig_type in ("breakout_up", "bounce_up", "sweep_down"):
        result = filter_signal_by_htf(
            _signal(sig_type, 100.0), "bullish", _empty_alignment(), zone_tol=1.0
        )
        assert result is True, f"{sig_type} should pass with bullish bias"


def test_filter_bullish_bias_blocks_bearish_signals():
    for sig_type in ("breakdown_down", "reject_down", "sweep_up"):
        result = filter_signal_by_htf(
            _signal(sig_type, 100.0), "bullish", _empty_alignment(), zone_tol=1.0
        )
        assert result is False, f"{sig_type} should be blocked by bullish bias"


def test_filter_bearish_bias_allows_bearish_signals():
    for sig_type in ("breakdown_down", "reject_down", "sweep_up"):
        result = filter_signal_by_htf(
            _signal(sig_type, 100.0), "bearish", _empty_alignment(), zone_tol=1.0
        )
        assert result is True, f"{sig_type} should pass with bearish bias"


def test_filter_bearish_bias_blocks_bullish_signals():
    for sig_type in ("breakout_up", "bounce_up", "sweep_down"):
        result = filter_signal_by_htf(
            _signal(sig_type, 100.0), "bearish", _empty_alignment(), zone_tol=1.0
        )
        assert result is False, f"{sig_type} should be blocked by bearish bias"


# ---------------------------------------------------------------------------
# filter_signal_by_htf — sweep-at-HTF override (Rule 3)
# ---------------------------------------------------------------------------


def test_filter_sweep_at_htf_zone_overrides_bearish_bias():
    """
    sweep_down is bullish-aligned; normally blocked by bearish bias.
    BUT: if it occurs at an HTF zone, Rule 3 overrides and allows it.
    """
    alignment = _alignment_with_htf_at(center=100.0)
    signal = _signal("sweep_down", zone_center=100.0)

    result = filter_signal_by_htf(signal, "bearish", alignment, zone_tol=1.0)
    assert result is True, "Sweep at HTF level should override bearish bias"


def test_filter_sweep_at_htf_zone_overrides_bullish_bias():
    """
    sweep_up is bearish-aligned; normally blocked by bullish bias.
    BUT: at an HTF zone, the stop-hunt reversal special case fires.
    """
    alignment = _alignment_with_htf_at(center=100.0)
    signal = _signal("sweep_up", zone_center=100.0)

    result = filter_signal_by_htf(signal, "bullish", alignment, zone_tol=1.0)
    assert result is True, "Sweep at HTF level should override bullish bias"


def test_filter_sweep_not_at_htf_zone_is_blocked():
    """
    Counter-trend sweep but NOT near any HTF zone → blocked normally.
    (sweep_down with bearish bias, HTF zone is far away at 200)
    """
    alignment = _alignment_with_htf_at(center=200.0)
    signal = _signal("sweep_down", zone_center=100.0)  # distance to HTF = 100

    result = filter_signal_by_htf(signal, "bearish", alignment, zone_tol=1.0)
    assert result is False, "Counter-trend sweep far from HTF should be blocked"


def test_filter_sweep_uses_linked_htf_zones_too():
    """
    The override also considers HTF zones that appear via the 'linked' list,
    not only via 'htf_only'.
    """
    htf_zone = _res(99.5, 100.5)   # center = 100.0
    ltf_zone = _res(99.5, 100.5)
    alignment = {
        "linked": [{"ltf": ltf_zone, "htf": htf_zone, "distance": 0.0}],
        "unlinked_ltf": [],
        "htf_only": [],
    }
    signal = _signal("sweep_up", zone_center=100.0)  # at HTF zone

    result = filter_signal_by_htf(signal, "bullish", alignment, zone_tol=1.0)
    assert result is True


def test_filter_sweep_no_htf_zones_at_all():
    """
    When alignment is completely empty (no HTF zones at all), the sweep
    override cannot fire; counter-trend sweep is blocked.
    """
    signal = _signal("sweep_down", zone_center=100.0)
    result = filter_signal_by_htf(signal, "bearish", _empty_alignment(), zone_tol=1.0)
    assert result is False


def test_filter_sweep_htf_proximity_uses_zone_tol():
    """
    The tol parameter controls how close the signal must be to the HTF zone.
    signal center = 100, HTF center = 102, distance = 2.0

    tol=1.0 → too far → counter-trend sweep blocked
    tol=3.0 → close enough → override fires
    """
    alignment = _alignment_with_htf_at(center=102.0)
    signal = _signal("sweep_down", zone_center=100.0)

    assert filter_signal_by_htf(signal, "bearish", alignment, zone_tol=1.0) is False
    assert filter_signal_by_htf(signal, "bearish", alignment, zone_tol=3.0) is True


# ---------------------------------------------------------------------------
# _near_any_htf_zone (unit-tested directly)
# ---------------------------------------------------------------------------


def test_near_any_htf_zone_true_via_htf_only():
    alignment = _alignment_with_htf_at(center=100.0)
    assert _near_any_htf_zone(100.0, alignment, tol=1.0) is True


def test_near_any_htf_zone_true_via_linked():
    htf_zone = _res(99.5, 100.5)
    alignment = {
        "linked": [{"ltf": _res(99.5, 100.5), "htf": htf_zone, "distance": 0.0}],
        "unlinked_ltf": [],
        "htf_only": [],
    }
    assert _near_any_htf_zone(100.0, alignment, tol=1.0) is True


def test_near_any_htf_zone_false_when_all_far():
    alignment = _alignment_with_htf_at(center=200.0)
    assert _near_any_htf_zone(100.0, alignment, tol=1.0) is False


def test_near_any_htf_zone_false_when_empty():
    assert _near_any_htf_zone(100.0, _empty_alignment(), tol=1.0) is False


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # align_zones — basic
        test_align_linked_when_within_tol,
        test_align_unlinked_when_outside_tol,
        test_align_boundary_exactly_at_tol,
        test_align_boundary_just_outside_tol,
        # align_zones — many-to-one / htf_only
        test_align_two_ltf_link_to_same_htf,
        test_align_htf_only_when_unmatched_htf,
        test_align_multiple_ltf_nearest_htf_selected,
        test_align_linked_sorted_by_distance,
        # align_zones — empty inputs
        test_align_empty_htf_all_ltf_unlinked,
        test_align_empty_ltf_all_htf_only,
        test_align_both_empty,
        # align_zones — validation
        test_align_raises_on_nonpositive_tol,
        # get_htf_bias
        test_get_htf_bias_bullish,
        test_get_htf_bias_bearish,
        test_get_htf_bias_uses_only_last_bar,
        # filter_signal_by_htf — basic rules
        test_filter_none_signal_returns_false,
        test_filter_neutral_bias_allows_any_signal,
        test_filter_bullish_bias_allows_bullish_signals,
        test_filter_bullish_bias_blocks_bearish_signals,
        test_filter_bearish_bias_allows_bearish_signals,
        test_filter_bearish_bias_blocks_bullish_signals,
        # filter_signal_by_htf — sweep override
        test_filter_sweep_at_htf_zone_overrides_bearish_bias,
        test_filter_sweep_at_htf_zone_overrides_bullish_bias,
        test_filter_sweep_not_at_htf_zone_is_blocked,
        test_filter_sweep_uses_linked_htf_zones_too,
        test_filter_sweep_no_htf_zones_at_all,
        test_filter_sweep_htf_proximity_uses_zone_tol,
        # _near_any_htf_zone
        test_near_any_htf_zone_true_via_htf_only,
        test_near_any_htf_zone_true_via_linked,
        test_near_any_htf_zone_false_when_all_far,
        test_near_any_htf_zone_false_when_empty,
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

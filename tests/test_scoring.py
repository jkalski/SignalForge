"""
Tests for backend.signals.scoring.score_setup.

Compatible with pytest (run: pytest tests/test_scoring.py -v)
and also executable standalone: python tests/test_scoring.py

Coverage
--------
_score_zone      — touch steps, recency thresholds, width/ATR buckets
_score_volume    — vol_ratio steps, None, warmup
_score_event     — all 6 event types + unknown
_score_ema       — four alignment combinations
_score_vwap      — session and anchored thresholds, None inputs
_score_mtf       — four bias/zone combinations
score_setup      — integration: perfect 100, low score, all-None optionals
score_setup      — reasons dict structure, score cap, pipeline compat
Validation       — atr14, vol_ratio, missing type key
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from backend.indicators.zones import Zone
from backend.signals.scoring import (
    score_setup,
    _score_zone,
    _score_volume,
    _score_event,
    _score_ema,
    _score_vwap,
    _score_mtf,
    _zone_age_days,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REF_TS = pd.Timestamp("2024-06-01")


def _zone(
    kind="resistance",
    low=99.0,
    high=101.0,
    touches=3,
    last_ts: pd.Timestamp = _REF_TS,
):
    return Zone(
        kind=kind,
        low=float(low),
        high=float(high),
        center=(low + high) / 2.0,
        touches=touches,
        first_ts=pd.Timestamp("2024-01-01"),
        last_ts=last_ts,
    )


def _event(event_type="bounce_up", zone_center=100.0):
    return {"type": event_type, "zone_center": zone_center, "touches": 3}


ATR = 1.0   # 1.0 makes width-to-ATR ratios easy to reason about


# ---------------------------------------------------------------------------
# _score_zone — touch steps
# ---------------------------------------------------------------------------


def test_zone_touches_2():
    z = _zone(touches=2, low=99, high=100)  # width=1 = 1*ATR (≥0.7 → precision=1)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 5


def test_zone_touches_3():
    z = _zone(touches=3, low=99, high=100)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 7


def test_zone_touches_4():
    z = _zone(touches=4, low=99, high=100)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 10


def test_zone_touches_5():
    z = _zone(touches=5, low=99, high=100)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 12


def test_zone_touches_6():
    z = _zone(touches=6, low=99, high=100)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 15


def test_zone_touches_above_6_capped():
    z = _zone(touches=10, low=99, high=100)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["touch_pts"] == 15


# ---------------------------------------------------------------------------
# _score_zone — recency thresholds
# ---------------------------------------------------------------------------


def test_zone_recency_0_days():
    z = _zone(last_ts=_REF_TS)
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_days"] == 0
    assert detail["recency_pts"] == 10


def test_zone_recency_5_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=5))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_days"] == 5
    assert detail["recency_pts"] == 10


def test_zone_recency_6_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=6))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_days"] == 6
    assert detail["recency_pts"] == 7


def test_zone_recency_20_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=20))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_pts"] == 7


def test_zone_recency_21_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=21))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_pts"] == 4


def test_zone_recency_60_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=60))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_pts"] == 4


def test_zone_recency_61_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=61))
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["recency_pts"] == 1


def test_zone_recency_none_ref_ts_defaults_to_0_days():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=500))
    detail = _score_zone(z, ATR, ref_ts=None)
    assert detail["recency_days"] == 0
    assert detail["recency_pts"] == 10   # assumed current


# ---------------------------------------------------------------------------
# _score_zone — width / ATR buckets
# ---------------------------------------------------------------------------


def test_zone_precision_narrow_below_03():
    z = _zone(low=99.7, high=100.0)     # width = 0.3 → ratio = 0.3/1.0 = 0.3 (< 0.3 False)
    # Actually 0.3 < 0.3 is False. Let's use 0.2:
    z2 = _zone(low=99.8, high=100.0)   # width = 0.2, ATR=1.0 → ratio = 0.2 < 0.3 → 5pts
    detail = _score_zone(z2, ATR, ref_ts=_REF_TS)
    assert detail["precision_pts"] == 5


def test_zone_precision_medium_03_to_07():
    z = _zone(low=99.5, high=100.0)    # width = 0.5, ATR=1.0 → ratio = 0.5 → 3pts
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["precision_pts"] == 3


def test_zone_precision_wide_above_07():
    z = _zone(low=99.0, high=100.0)    # width = 1.0, ATR=1.0 → ratio = 1.0 ≥ 0.7 → 1pt
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["precision_pts"] == 1


def test_zone_width_zero_is_most_precise():
    """A single-price zone (width=0) should score 5 precision pts."""
    z = _zone(low=100.0, high=100.0)   # width=0 → ratio=0 < 0.3 → 5pts
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["precision_pts"] == 5


def test_zone_total_capped_at_30():
    """Even the highest inputs cannot exceed 30."""
    z = _zone(touches=99, low=100.0, high=100.0, last_ts=_REF_TS)  # max touch + zero width
    detail = _score_zone(z, ATR, ref_ts=_REF_TS)
    assert detail["score"] <= 30
    assert detail["max"] == 30


# ---------------------------------------------------------------------------
# _score_volume
# ---------------------------------------------------------------------------


def test_volume_none_is_zero():
    assert _score_volume(None)["score"] == 0


def test_volume_below_average():
    assert _score_volume(0.8)["score"] == 0


def test_volume_at_1x():
    assert _score_volume(1.0)["score"] == 5


def test_volume_at_1_5x():
    assert _score_volume(1.5)["score"] == 10


def test_volume_between_1_5_and_2():
    assert _score_volume(1.9)["score"] == 10


def test_volume_at_2x():
    assert _score_volume(2.0)["score"] == 15


def test_volume_at_3x():
    assert _score_volume(3.0)["score"] == 20


def test_volume_above_3x():
    assert _score_volume(5.0)["score"] == 20


def test_volume_max_is_20():
    assert _score_volume(100.0)["max"] == 20


# ---------------------------------------------------------------------------
# _score_event
# ---------------------------------------------------------------------------


def test_event_sweep_up():
    assert _score_event("sweep_up")["score"] == 10


def test_event_sweep_down():
    assert _score_event("sweep_down")["score"] == 10


def test_event_bounce_up():
    assert _score_event("bounce_up")["score"] == 8


def test_event_reject_down():
    assert _score_event("reject_down")["score"] == 8


def test_event_breakout_up():
    assert _score_event("breakout_up")["score"] == 6


def test_event_breakdown_down():
    assert _score_event("breakdown_down")["score"] == 6


def test_event_unknown_is_zero():
    assert _score_event("foobar")["score"] == 0


def test_event_max_is_10():
    assert _score_event("sweep_up")["max"] == 10


# ---------------------------------------------------------------------------
# _score_ema
# ---------------------------------------------------------------------------


def test_ema_both_aligned():
    assert _score_ema(True, True)["score"] == 15


def test_ema_ltf_only():
    assert _score_ema(True, None)["score"] == 8
    assert _score_ema(True, False)["score"] == 8


def test_ema_htf_only():
    assert _score_ema(False, True)["score"] == 4


def test_ema_neither():
    assert _score_ema(False, False)["score"] == 0
    assert _score_ema(False, None)["score"] == 0


def test_ema_max_is_15():
    assert _score_ema(True, True)["max"] == 15


# ---------------------------------------------------------------------------
# _score_vwap
# ---------------------------------------------------------------------------


def test_vwap_both_none():
    d = _score_vwap(None, None)
    assert d["score"] == 0
    assert d["session_pts"] == 0
    assert d["anchored_pts"] == 0


def test_vwap_session_very_close():
    """session dist ≤ 0.2 % → 10 pts."""
    d = _score_vwap(0.15, None)
    assert d["session_pts"] == 10
    assert d["anchored_pts"] == 0
    assert d["score"] == 10


def test_vwap_session_moderate():
    """0.2 < dist ≤ 0.5 % → 7 pts."""
    d = _score_vwap(0.35, None)
    assert d["session_pts"] == 7


def test_vwap_session_far():
    """0.5 < dist ≤ 1.0 % → 4 pts."""
    d = _score_vwap(0.8, None)
    assert d["session_pts"] == 4


def test_vwap_session_too_far():
    """dist > 1.0 % → 0 pts."""
    d = _score_vwap(1.5, None)
    assert d["session_pts"] == 0


def test_vwap_anchored_very_close():
    """anchored dist ≤ 0.2 % → 5 pts."""
    d = _score_vwap(None, 0.1)
    assert d["anchored_pts"] == 5
    assert d["score"] == 5


def test_vwap_anchored_moderate():
    """0.2 < dist ≤ 0.5 % → 3 pts."""
    d = _score_vwap(None, 0.4)
    assert d["anchored_pts"] == 3


def test_vwap_anchored_far():
    """0.5 < dist ≤ 1.0 % → 1 pt."""
    d = _score_vwap(None, 0.9)
    assert d["anchored_pts"] == 1


def test_vwap_anchored_too_far():
    d = _score_vwap(None, 2.0)
    assert d["anchored_pts"] == 0


def test_vwap_both_contribute():
    """Both session and anchored contribute independently."""
    d = _score_vwap(0.1, 0.1)
    assert d["session_pts"] == 10
    assert d["anchored_pts"] == 5
    assert d["score"] == 15


def test_vwap_max_is_15():
    assert _score_vwap(0.0, 0.0)["max"] == 15


# ---------------------------------------------------------------------------
# _score_mtf
# ---------------------------------------------------------------------------


def test_mtf_both():
    assert _score_mtf(True, True)["score"] == 10


def test_mtf_near_htf_only():
    assert _score_mtf(False, True)["score"] == 7


def test_mtf_bias_only():
    assert _score_mtf(True, False)["score"] == 4


def test_mtf_neither():
    assert _score_mtf(False, False)["score"] == 0


def test_mtf_max_is_10():
    assert _score_mtf(True, True)["max"] == 10


# ---------------------------------------------------------------------------
# score_setup — integration
# ---------------------------------------------------------------------------


def _perfect_zone():
    """Zone that earns the maximum sub-score (30 pts)."""
    return _zone(
        touches=6,          # 15 pts
        low=100.0,
        high=100.0,         # width=0 → precision=5 pts
        last_ts=_REF_TS,    # age=0 → recency=10 pts
    )


def test_score_setup_perfect_100():
    """All inputs at maximum → total score = 100."""
    result = score_setup(
        event=_event("sweep_down"),     # 10 pts
        zone=_perfect_zone(),           # 30 pts
        atr14=1.0,
        vol_ratio=4.0,                  # 20 pts
        ltf_trend_aligned=True,
        htf_trend_aligned=True,         # 15 pts
        vwap_session_dist_pct=0.1,      # 10 pts
        vwap_anchored_dist_pct=0.1,     # 5 pts  → vwap total 15
        htf_bias_aligned=True,
        near_htf_zone=True,             # 10 pts
        ref_ts=_REF_TS,
    )
    assert result["score"] == 100


def test_score_setup_low_score():
    """
    Weak-quality setup: 2-touch stale wide zone, no volume, breakout only,
    no EMA, no VWAP, no MTF.

    Expected sub-scores:
      zone   : touches=2(5) + recency_100d(1) + width_1ATR(1) = 7
      volume : vol_ratio=0.5 → 0
      event  : breakout_up → 6
      ema    : neither → 0
      vwap   : both None → 0
      mtf    : neither → 0
      total  : 13
    """
    zone = _zone(
        touches=2,
        low=99.0,
        high=100.0,                              # width=1.0 = 1.0 ATR → 1pt
        last_ts=_REF_TS - pd.Timedelta(days=100),  # stale → 1pt recency
    )
    result = score_setup(
        event=_event("breakout_up"),
        zone=zone,
        atr14=1.0,
        vol_ratio=0.5,
        ltf_trend_aligned=False,
        htf_trend_aligned=False,
        ref_ts=_REF_TS,
    )
    assert result["score"] == 13


def test_score_setup_all_optional_none():
    """No optional inputs provided → still returns a valid result."""
    result = score_setup(
        event=_event("bounce_up"),
        zone=_zone(touches=3),
        atr14=1.0,
    )
    assert 0 <= result["score"] <= 100
    assert result["reasons"]["volume"]["score"] == 0    # no vol_ratio
    assert result["reasons"]["vwap"]["score"] == 0      # no VWAP inputs
    assert result["reasons"]["mtf"]["score"] == 0       # both False by default
    assert result["reasons"]["ema"]["score"] == 0       # both False by default


def test_score_setup_score_capped_at_100():
    """
    Even if individual components could theoretically overflow the max
    (e.g. 15+10+5 = 30 for zone), the total is always ≤ 100.
    Score with all maximal inputs must be exactly 100, not 101+.
    """
    result = score_setup(
        event=_event("sweep_up"),
        zone=_perfect_zone(),
        atr14=1.0,
        vol_ratio=999.0,
        ltf_trend_aligned=True,
        htf_trend_aligned=True,
        vwap_session_dist_pct=0.0,
        vwap_anchored_dist_pct=0.0,
        htf_bias_aligned=True,
        near_htf_zone=True,
        ref_ts=_REF_TS,
    )
    assert result["score"] <= 100


def test_score_setup_reasons_structure():
    """reasons dict contains all expected keys with 'score' and 'max' sub-keys."""
    result = score_setup(
        event=_event("bounce_up"),
        zone=_zone(),
        atr14=1.0,
        ref_ts=_REF_TS,
    )
    reasons = result["reasons"]
    expected_keys = {"zone", "volume", "event", "ema", "vwap", "mtf"}
    assert set(reasons.keys()) == expected_keys

    for key in expected_keys:
        assert "score" in reasons[key], f"'score' missing from reasons['{key}']"
        assert "max"   in reasons[key], f"'max'   missing from reasons['{key}']"


def test_score_setup_reasons_max_sums_to_100():
    """The 'max' values across all dimensions must sum to exactly 100."""
    result = score_setup(
        event=_event("bounce_up"),
        zone=_zone(),
        atr14=1.0,
    )
    total_max = sum(v["max"] for v in result["reasons"].values())
    assert total_max == 100


def test_score_setup_intermediate():
    """
    Manually computed intermediate case.

    zone: 3 touches(7) + 10d recency(7) + 0.5ATR width(3) = 17
    vol_ratio=2.0 → 15
    event=bounce_up → 8
    ltf=True, htf=None → 8
    session_dist=0.35 → 7 pts, anchored=None → 0 → total vwap=7
    near_htf=True, bias=False → 7

    total = 17+15+8+8+7+7 = 62
    """
    zone = _zone(
        touches=3,
        low=99.75,
        high=100.25,                              # width=0.5 → 0.5/1.0=0.5 ATR → 3pts
        last_ts=_REF_TS - pd.Timedelta(days=10),  # recency 7pts
    )
    result = score_setup(
        event=_event("bounce_up"),
        zone=zone,
        atr14=1.0,
        vol_ratio=2.0,
        ltf_trend_aligned=True,
        htf_trend_aligned=None,
        vwap_session_dist_pct=0.35,
        vwap_anchored_dist_pct=None,
        htf_bias_aligned=False,
        near_htf_zone=True,
        ref_ts=_REF_TS,
    )
    assert result["score"] == 62, f"Expected 62, got {result['score']}"


def test_score_setup_pipeline_compatible_fields():
    """
    Verify that common pipeline output fields map cleanly to scorer inputs.

    pipeline['vol_ratio']   → vol_ratio
    pipeline['ema_confirms'] → ltf_trend_aligned
    pipeline['atr_14']       → atr14
    """
    # Simulate pipeline output subset
    pipeline_out = {
        "vol_ratio":    2.5,
        "ema_confirms": True,
        "atr_14":       1.5,
    }
    result = score_setup(
        event=_event("breakout_up"),
        zone=_zone(),
        atr14=pipeline_out["atr_14"],
        vol_ratio=pipeline_out["vol_ratio"],
        ltf_trend_aligned=pipeline_out["ema_confirms"],
        ref_ts=_REF_TS,
    )
    # Vol ratio 2.5 → 15pts (2.0–3.0 bracket)
    assert result["reasons"]["volume"]["score"] == 15
    # EMA: ltf only → 8pts
    assert result["reasons"]["ema"]["score"] == 8


# ---------------------------------------------------------------------------
# _zone_age_days
# ---------------------------------------------------------------------------


def test_zone_age_days_zero():
    z = _zone(last_ts=_REF_TS)
    assert _zone_age_days(z, _REF_TS) == 0


def test_zone_age_days_positive():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=15))
    assert _zone_age_days(z, _REF_TS) == 15


def test_zone_age_days_none_ref():
    z = _zone(last_ts=_REF_TS - pd.Timedelta(days=500))
    assert _zone_age_days(z, None) == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_raises_on_missing_type_key():
    try:
        score_setup(event={"zone_center": 100.0}, zone=_zone(), atr14=1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "type" in str(e)


def test_raises_on_nonpositive_atr():
    try:
        score_setup(event=_event(), zone=_zone(), atr14=0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "atr14" in str(e)

    try:
        score_setup(event=_event(), zone=_zone(), atr14=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "atr14" in str(e)


def test_raises_on_negative_vol_ratio():
    try:
        score_setup(event=_event(), zone=_zone(), atr14=1.0, vol_ratio=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "vol_ratio" in str(e)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # zone — touches
        test_zone_touches_2,
        test_zone_touches_3,
        test_zone_touches_4,
        test_zone_touches_5,
        test_zone_touches_6,
        test_zone_touches_above_6_capped,
        # zone — recency
        test_zone_recency_0_days,
        test_zone_recency_5_days,
        test_zone_recency_6_days,
        test_zone_recency_20_days,
        test_zone_recency_21_days,
        test_zone_recency_60_days,
        test_zone_recency_61_days,
        test_zone_recency_none_ref_ts_defaults_to_0_days,
        # zone — precision
        test_zone_precision_narrow_below_03,
        test_zone_precision_medium_03_to_07,
        test_zone_precision_wide_above_07,
        test_zone_width_zero_is_most_precise,
        test_zone_total_capped_at_30,
        # volume
        test_volume_none_is_zero,
        test_volume_below_average,
        test_volume_at_1x,
        test_volume_at_1_5x,
        test_volume_between_1_5_and_2,
        test_volume_at_2x,
        test_volume_at_3x,
        test_volume_above_3x,
        test_volume_max_is_20,
        # event type
        test_event_sweep_up,
        test_event_sweep_down,
        test_event_bounce_up,
        test_event_reject_down,
        test_event_breakout_up,
        test_event_breakdown_down,
        test_event_unknown_is_zero,
        test_event_max_is_10,
        # ema
        test_ema_both_aligned,
        test_ema_ltf_only,
        test_ema_htf_only,
        test_ema_neither,
        test_ema_max_is_15,
        # vwap
        test_vwap_both_none,
        test_vwap_session_very_close,
        test_vwap_session_moderate,
        test_vwap_session_far,
        test_vwap_session_too_far,
        test_vwap_anchored_very_close,
        test_vwap_anchored_moderate,
        test_vwap_anchored_far,
        test_vwap_anchored_too_far,
        test_vwap_both_contribute,
        test_vwap_max_is_15,
        # mtf
        test_mtf_both,
        test_mtf_near_htf_only,
        test_mtf_bias_only,
        test_mtf_neither,
        test_mtf_max_is_10,
        # integration
        test_score_setup_perfect_100,
        test_score_setup_low_score,
        test_score_setup_all_optional_none,
        test_score_setup_score_capped_at_100,
        test_score_setup_reasons_structure,
        test_score_setup_reasons_max_sums_to_100,
        test_score_setup_intermediate,
        test_score_setup_pipeline_compatible_fields,
        # zone age helper
        test_zone_age_days_zero,
        test_zone_age_days_positive,
        test_zone_age_days_none_ref,
        # validation
        test_raises_on_missing_type_key,
        test_raises_on_nonpositive_atr,
        test_raises_on_negative_vol_ratio,
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

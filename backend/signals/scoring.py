"""
Confluence scoring system for trading setups.

Produces a single integer score in [0, 100] that aggregates six independent
evidence dimensions.  Each dimension is scored on a fixed sub-range whose
maximum sums to exactly 100 — no renormalisation is needed.

Weight budget
-------------
  Zone strength    30 pts   (touches, recency, precision)
  Volume           20 pts   (vol / vol_sma ratio)
  Event type       10 pts   (sweep > structural > breakout)
  EMA confirmation 15 pts   (LTF and optional HTF trend alignment)
  VWAP confluence  15 pts   (session VWAP 10 + anchored VWAP 5)
  MTF alignment    10 pts   (HTF bias + HTF zone proximity)
  ───────────────────────
  Maximum          100 pts

Persistence
-----------
``score_setup()`` is pure Python — it never touches the database.  Callers
are responsible for storing the result:

  * ``result["score"]``           → ``Setup.score``  (existing Numeric column)
  * ``json.dumps(result["reasons"])`` → ``Signal.context_snapshot``
                                         (existing Text column)

If the breakdown also needs to be persisted against a ``Setup`` row, add a
``context_snapshot TEXT`` column to the ``setups`` table via an Alembic
migration (not included here; only one line of model code is required).

Distance convention
-------------------
All ``*_dist_pct`` parameters follow the same convention as the pipeline's
``distance_pct`` field: values are **actual percentage points**, where
``1.0`` means 1 % (not 0.01).  Compute as:

    abs(close - vwap) / close * 100
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

from backend.indicators.zones import Zone


# ---------------------------------------------------------------------------
# Weight constants — must sum to 100
# ---------------------------------------------------------------------------

_W_ZONE   = 30
_W_VOLUME = 20
_W_EVENT  = 10
_W_EMA    = 15
_W_VWAP   = 15
_W_MTF    = 10

assert _W_ZONE + _W_VOLUME + _W_EVENT + _W_EMA + _W_VWAP + _W_MTF == 100

# ---------------------------------------------------------------------------
# Event-type sets (module-level constants — avoids re-creating sets on every call)
# ---------------------------------------------------------------------------

_SWEEP_EVENTS       = frozenset({"sweep_up", "sweep_down"})
_STRUCTURAL_EVENTS  = frozenset({"bounce_up", "reject_down"})
_BREAKOUT_EVENTS    = frozenset({"breakout_up", "breakdown_down"})
_BOS_EVENTS         = frozenset({"bos_up", "bos_down", "choch_up", "choch_down"})
_FVG_EVENTS         = frozenset({"fvg_long", "fvg_short"})
_VWAP_EVENTS        = frozenset({"vwap_reclaim_long", "vwap_reclaim_short"})
_ORB_EVENTS         = frozenset({"orb_long", "orb_short"})
_GAP_FADE_EVENTS    = frozenset({"gap_fade_long", "gap_fade_short"})
_GAP_GO_EVENTS      = frozenset({"gap_go_long", "gap_go_short"})
_OUTSIDE_BAR_EVENTS = frozenset({"outside_bar_long", "outside_bar_short"})
_INSIDE_BAR_EVENTS  = frozenset({"inside_bar_long", "inside_bar_short"})
_DOUBLE_IB_EVENTS   = frozenset({"double_inside_bar_long", "double_inside_bar_short"})
_ATH_EVENTS         = frozenset({"ath_breakout_long"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_setup(
    event: Dict[str, Any],
    zone: Zone,
    atr14: float,
    vol_ratio: Optional[float] = None,
    ltf_trend_aligned: bool = False,
    htf_trend_aligned: Optional[bool] = None,
    vwap_session_dist_pct: Optional[float] = None,
    vwap_anchored_dist_pct: Optional[float] = None,
    htf_bias_aligned: bool = False,
    near_htf_zone: bool = False,
    ref_ts: Optional[pd.Timestamp] = None,
    rsi_divergence_aligned: bool = False,
) -> Dict[str, Any]:
    """
    Compute a 0-100 confluence score for a single setup.

    Parameters
    ----------
    event : dict
        Output of ``detect_breakout_or_bounce`` or ``detect_liquidity_sweeps``.
        Must contain ``"type"`` (str) and ``"zone_center"`` (float) keys.
    zone : Zone
        The structural zone that triggered the event.  Used for touches,
        recency (``last_ts``), and width.
    atr14 : float
        14-period ATR of the lower-timeframe candles.  Must be > 0.  Used
        to normalise zone width into ATR multiples.
    vol_ratio : float or None
        ``volume / vol_sma`` for the triggering bar.  ``None`` when the
        volume SMA warmup period has not yet completed.  Values < 0 raise
        ``ValueError``.
    ltf_trend_aligned : bool, default False
        True when the LTF EMA50 bias agrees with the event direction.
        Equivalent to ``ema_confirms`` in the existing pipeline output.
    htf_trend_aligned : bool or None, default None
        True / False from ``get_htf_bias``.  ``None`` means HTF data was
        not available; no penalty is applied.
    vwap_session_dist_pct : float or None, default None
        ``abs(close - vwap_session) / close * 100``.  ``None`` when session
        VWAP was not computed (e.g. daily timeframe or missing volume).
    vwap_anchored_dist_pct : float or None, default None
        ``abs(close - vwap_anchored) / close * 100``.  ``None`` when no
        anchored VWAP was computed.
    htf_bias_aligned : bool, default False
        True when the LTF event direction agrees with the HTF EMA50 bias.
        Computed via ``filter_signal_by_htf`` or equivalent.
    near_htf_zone : bool, default False
        True when the event's zone center falls within tolerance of at least
        one HTF structural zone (output of ``_near_any_htf_zone``).
    ref_ts : pd.Timestamp or None, default None
        Timestamp of the triggering bar.  Used to compute zone recency (age
        in calendar days since ``zone.last_ts``).  When ``None``, age is
        assumed to be 0 days (maximum recency score).
    rsi_divergence_aligned : bool, default False
        True when RSI divergence (bullish/bearish) agrees with signal direction.
        Awards a bonus 8 pts (additive; final score capped at 100).

    Returns
    -------
    dict
        ``"score"`` : int — final score in [0, 100].
        ``"reasons"`` : dict — per-dimension breakdown with ``"score"`` and
        ``"max"`` keys plus diagnostic fields for debugging.

    Raises
    ------
    ValueError
        If ``atr14 <= 0``, ``event`` is missing the ``"type"`` key, or
        ``vol_ratio`` is negative.
    """
    _validate(event, atr14, vol_ratio)

    zone_detail   = _score_zone(zone, atr14, ref_ts)
    volume_detail = _score_volume(vol_ratio)
    event_detail  = _score_event(event["type"])
    ema_detail    = _score_ema(ltf_trend_aligned, htf_trend_aligned)
    vwap_detail   = _score_vwap(vwap_session_dist_pct, vwap_anchored_dist_pct)
    mtf_detail    = _score_mtf(htf_bias_aligned, near_htf_zone)
    div_detail    = _score_divergence(rsi_divergence_aligned)

    raw = (
        zone_detail["score"]
        + volume_detail["score"]
        + event_detail["score"]
        + ema_detail["score"]
        + vwap_detail["score"]
        + mtf_detail["score"]
        + div_detail["score"]
    )

    return {
        "score": min(raw, 100),
        "reasons": {
            "zone":        zone_detail,
            "volume":      volume_detail,
            "event":       event_detail,
            "ema":         ema_detail,
            "vwap":        vwap_detail,
            "mtf":         mtf_detail,
            "divergence":  div_detail,
        },
    }


# ---------------------------------------------------------------------------
# Per-dimension scorers
# ---------------------------------------------------------------------------


def _score_zone(zone: Zone, atr14: float, ref_ts: Optional[pd.Timestamp]) -> Dict:
    """Zone strength: touches (15) + recency (10) + precision (5) = 30 max."""

    # --- Touches (0–15) ---------------------------------------------------
    # Linear scale capped at 6 touches; each touch beyond the first adds signal.
    touches = zone.touches
    if touches >= 6:
        touch_pts = 15
    elif touches >= 5:
        touch_pts = 12
    elif touches >= 4:
        touch_pts = 10
    elif touches >= 3:
        touch_pts = 7
    else:
        touch_pts = 5  # minimum (2 touches required by build_zones_from_pivots)

    # --- Recency (0–10) ---------------------------------------------------
    age_days = _zone_age_days(zone, ref_ts)
    if age_days <= 5:
        recency_pts = 10
    elif age_days <= 20:
        recency_pts = 7
    elif age_days <= 60:
        recency_pts = 4
    else:
        recency_pts = 1  # stale but still structurally relevant

    # --- Width / precision (0–5) ------------------------------------------
    # Narrow zones (small ATR multiple) produce cleaner entries and stops.
    width_ratio = zone.width / atr14  # dimensionless; 1.0 = exactly one ATR wide
    if width_ratio < 0.3:
        precision_pts = 5
    elif width_ratio < 0.7:
        precision_pts = 3
    else:
        precision_pts = 1

    total = min(touch_pts + recency_pts + precision_pts, _W_ZONE)
    return {
        "score":          total,
        "max":            _W_ZONE,
        "touches":        touches,
        "touch_pts":      touch_pts,
        "recency_days":   age_days,
        "recency_pts":    recency_pts,
        "width_atr_ratio": round(width_ratio, 3),
        "precision_pts":  precision_pts,
    }


def _score_volume(vol_ratio: Optional[float]) -> Dict:
    """
    Volume magnitude: 20 pts for ≥ 3× average, scaling down to 0 pts below average.

    Institutional activity shows as large vol_ratio spikes — this is the
    single strongest mechanical confirmation of price action.
    """
    if vol_ratio is None or not math.isfinite(vol_ratio):
        pts = 0
    elif vol_ratio >= 3.0:
        pts = 20   # strong institutional footprint
    elif vol_ratio >= 2.0:
        pts = 15
    elif vol_ratio >= 1.5:
        pts = 10
    elif vol_ratio >= 1.0:
        pts = 5    # average volume — minor confirmation only
    else:
        pts = 0    # below-average volume — no confirmation

    return {
        "score":     pts,
        "max":       _W_VOLUME,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
    }


def _score_event(event_type: str) -> Dict:
    """
    Event type base score (10 pts max).

    Sweep reversals rank highest: they offer the tightest invalidation
    (stop just beyond the sweep wick) and the cleanest R:R.
    Structural tests (bounce / rejection) rank second.
    Confirmed breakouts rank third — valid but wider stops and more false
    positives in choppy conditions.
    """
    if event_type in _SWEEP_EVENTS:
        pts = 10
    elif event_type in _ATH_EVENTS:
        pts = 9   # uncharted territory — no overhead supply, momentum can exceed targets
    elif event_type in _STRUCTURAL_EVENTS:
        pts = 8
    elif event_type in _GAP_FADE_EVENTS:
        pts = 8   # gap fades are high-probability mean-reversion setups
    elif event_type in _DOUBLE_IB_EVENTS:
        pts = 9   # 2+ inside bars = stronger consolidation, more explosive breakout
    elif event_type in _BREAKOUT_EVENTS or event_type in _GAP_GO_EVENTS:
        pts = 6
    elif event_type in _BOS_EVENTS or event_type in _FVG_EVENTS:
        pts = 7   # strong structural/imbalance signals
    elif event_type in _OUTSIDE_BAR_EVENTS or event_type in _INSIDE_BAR_EVENTS:
        pts = 7   # engulfing / single inside bar breakout — pattern-based confirmation
    elif event_type in _VWAP_EVENTS or event_type in _ORB_EVENTS:
        pts = 5   # session-context signals — valid but lower base confidence
    else:
        pts = 0

    return {
        "score":      pts,
        "max":        _W_EVENT,
        "event_type": event_type,
    }


def _score_ema(ltf_aligned: bool, htf_aligned: Optional[bool]) -> Dict:
    """
    EMA trend confirmation (15 pts max).

    Both timeframes aligned is the gold standard.  LTF-only alignment
    still earns more than half: we may simply not have HTF data.
    """
    if ltf_aligned and htf_aligned is True:
        pts = 15   # dual-timeframe trend agreement
    elif ltf_aligned:
        pts = 8    # LTF confirmed; HTF absent or counter
    elif htf_aligned is True:
        pts = 4    # HTF trend but LTF setup not yet aligned
    else:
        pts = 0

    return {
        "score":        pts,
        "max":          _W_EMA,
        "ltf_aligned":  ltf_aligned,
        "htf_aligned":  htf_aligned,
    }


def _score_vwap(
    session_dist_pct: Optional[float],
    anchored_dist_pct: Optional[float],
) -> Dict:
    """
    VWAP confluence (15 pts max = session 10 + anchored 5).

    VWAP acts as an intraday fair-value magnet.  Events that occur at or
    near VWAP carry stronger mean-reversion / momentum probability because
    VWAP is widely watched by institutional algo execution desks.

    Distance thresholds (actual percent, e.g. 0.5 means 0.5 %):
      session VWAP : ≤ 0.2 % → 10,  ≤ 0.5 % → 7,  ≤ 1.0 % → 4,  > 1.0 % → 0
      anchored VWAP: ≤ 0.2 % →  5,  ≤ 0.5 % → 3,  ≤ 1.0 % → 1,  > 1.0 % → 0
    """

    def _session_pts(d: Optional[float]) -> int:
        if d is None or not math.isfinite(d):
            return 0
        if d <= 0.2:
            return 10
        if d <= 0.5:
            return 7
        if d <= 1.0:
            return 4
        return 0

    def _anchored_pts(d: Optional[float]) -> int:
        if d is None or not math.isfinite(d):
            return 0
        if d <= 0.2:
            return 5
        if d <= 0.5:
            return 3
        if d <= 1.0:
            return 1
        return 0

    s_pts = _session_pts(session_dist_pct)
    a_pts = _anchored_pts(anchored_dist_pct)
    total = min(s_pts + a_pts, _W_VWAP)

    return {
        "score":               total,
        "max":                 _W_VWAP,
        "session_dist_pct":    round(session_dist_pct, 4)  if session_dist_pct  is not None else None,
        "session_pts":         s_pts,
        "anchored_dist_pct":   round(anchored_dist_pct, 4) if anchored_dist_pct is not None else None,
        "anchored_pts":        a_pts,
    }


def _score_mtf(htf_bias_aligned: bool, near_htf_zone: bool) -> Dict:
    """
    Multi-timeframe alignment (10 pts max).

    Near an HTF structural level earns more than pure bias agreement:
    structural confluence is a harder, more actionable filter than trend.
    """
    if htf_bias_aligned and near_htf_zone:
        pts = 10
    elif near_htf_zone:
        pts = 7
    elif htf_bias_aligned:
        pts = 4
    else:
        pts = 0

    return {
        "score":            pts,
        "max":              _W_MTF,
        "htf_bias_aligned": htf_bias_aligned,
        "near_htf_zone":    near_htf_zone,
    }


def _score_divergence(aligned: bool) -> Dict:
    """
    RSI divergence confluence bonus (8 pts max).

    When RSI divergence is present AND agrees with the signal direction, it
    represents a meaningful momentum confirmation that deserves extra weight.
    This is additive: the total score can reach 108 pts but is capped at 100
    by score_setup().

    8 pts chosen to match the MTF bucket weight — divergence is roughly as
    meaningful as HTF zone confluence but less universal (depends on pivot data).
    """
    pts = 8 if aligned else 0
    return {
        "score":   pts,
        "max":     8,
        "aligned": aligned,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zone_age_days(zone: Zone, ref_ts: Optional[pd.Timestamp]) -> int:
    """Calendar days between the zone's last pivot and the triggering bar."""
    if ref_ts is None:
        return 0
    try:
        delta = pd.Timestamp(ref_ts) - pd.Timestamp(zone.last_ts)
        return max(int(delta.days), 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    event: Dict[str, Any],
    atr14: float,
    vol_ratio: Optional[float],
) -> None:
    if "type" not in event:
        raise ValueError("event dict must contain a 'type' key.")
    if not (atr14 > 0 and math.isfinite(atr14)):
        raise ValueError(f"atr14 must be a finite positive number, got {atr14!r}.")
    if vol_ratio is not None and vol_ratio < 0:
        raise ValueError(f"vol_ratio must be >= 0 or None, got {vol_ratio!r}.")

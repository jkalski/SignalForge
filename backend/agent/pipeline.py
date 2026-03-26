"""
Institutional-lite signal pipeline.

Orchestrates the full analysis chain for a single symbol's candle history.
All signal logic operates exclusively on the normalised candle schema
(ts, open, high, low, close, volume).  Provider differences belong in
ingestion/normalisation — zero provider-specific branching lives here.

Step order
----------
  1.  compute_features          EMA20/50, ATR14, RSI14
  2.  Build DataFrame           normalise dtypes
  3.  Data-quality assessment   volume_ok, gaps_detected, bars_count
  4.  find_pivots               swing high / low detection
  5.  add_volume_signals        20-bar vol SMA + spike flag
  6.  add_trend_filter          EMA trend bias columns
  7.  LTF zone clustering       S/R zones on lower timeframe
  8.  HTF zone clustering       (optional) S/R zones + bias on higher TF
  9.  VWAP                      session reset + anchored (volume-gated)
 10.  Nearby-zone selection     closest N zones only (avoids O(n*zones))
 11.  Event detection           sweeps first → bounces → breakouts
 12.  Gate checks               volume, EMA, MTF alignment
 13.  Confluence scoring        score_setup → 0-100
 14.  Position sizing           ATR-based stop/target

Backward compatibility
----------------------
All fields present in the original run_structure_pipeline output are
preserved with unchanged semantics.  New fields are purely additive.
The single required change: an optional `htf_candles` parameter is added
(default None → legacy behaviour, no HTF enrichment).

Signal status
-------------
  "active"    — all gates pass: event + vol_spike + ema_confirms + MTF
  "watchlist" — event detected but a gate failed (missing vol, counter-trend)
  None        — no event on the last bar (pipeline returns None)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.features.indicators import compute_features
from backend.indicators.pivots import find_pivots, adaptive_pivot_order
from backend.indicators.volume import add_volume_signals
from backend.indicators.trend import add_trend_filter
from backend.indicators.zones import Zone, build_zones_from_pivots, invalidate_broken_zones
from backend.indicators.vwap import (
    add_session_vwap,
    add_anchored_vwap,
    get_last_major_pivot_index,
)
from backend.indicators.mtf import (
    align_zones,
    get_htf_bias,
    filter_signal_by_htf,
    _near_any_htf_zone,
)
from backend.signals.structure_signals import detect_breakout_or_bounce
from backend.signals.liquidity import detect_liquidity_sweeps
from backend.signals.scoring import score_setup
from backend.indicators.daily_levels import get_daily_levels
from backend.signals.fvg import detect_fvg
from backend.signals.vwap_signals import detect_vwap_reclaim
from backend.signals.orb import detect_orb
from backend.signals.market_structure import detect_bos
from backend.signals.divergence import detect_rsi_divergence
from backend.signals.gap import detect_gap
from backend.signals.bar_patterns import detect_inside_bar, detect_outside_bar, detect_ath_breakout
from backend.ml.predictor import get_predictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BARS       = 52    # EMA50 needs 50+ bars; hard minimum
_MIN_HTF_BARS   = 40    # minimum useful HTF history
_PIVOT_ORDER    = 3     # ±3 bars for swing detection
_ZONE_ATR_MULT  = 0.5   # zone_tol = ATR14 * 0.5
_MIN_TOUCHES    = 2     # discard zones with fewer pivot touches
_MAX_NEARBY     = 6     # zones passed to event detectors (perf cap)
_SWEEP_LOOKBACK = 2     # bars back for sweep wick search

ATR_STOP_MULT   = 1.5
ATR_TARGET_MULT = 3.0
R_MULTIPLE      = round(ATR_TARGET_MULT / ATR_STOP_MULT, 2)  # 2.0

# Event type → trade direction (includes sweep events)
_DIRECTION: Dict[str, str] = {
    "breakout_up":         "long",
    "bounce_up":           "long",
    "sweep_down":          "long",    # swept below support, closed above → long
    "breakdown_down":      "short",
    "reject_down":         "short",
    "sweep_up":            "short",   # swept above resistance, closed below → short
    # New signal types
    "fvg_long":            "long",
    "fvg_short":           "short",
    "vwap_reclaim_long":   "long",
    "vwap_reclaim_short":  "short",
    "orb_long":            "long",
    "orb_short":           "short",
    "bos_up":              "long",
    "bos_down":            "short",
    "choch_up":            "long",
    "choch_down":          "short",
    "gap_fade_long":            "long",
    "gap_fade_short":           "short",
    "gap_go_long":              "long",
    "gap_go_short":             "short",
    "ath_breakout_long":        "long",
    "inside_bar_long":          "long",
    "inside_bar_short":         "short",
    "double_inside_bar_long":   "long",
    "double_inside_bar_short":  "short",
    "outside_bar_long":         "long",
    "outside_bar_short":        "short",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_structure_pipeline(
    candles: List[Dict[str, Any]],
    symbol: str = "",
    timeframe: str = "",
    source: Optional[str] = None,
    htf_candles: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the institutional-lite pipeline on one symbol's candle history.

    Parameters
    ----------
    candles : list of dicts
        LTF OHLCV rows sorted ascending by ts.
        Required keys: ts, open, high, low, close, volume.
        volume may be None / 0 — all volume-dependent steps degrade gracefully.
    symbol, timeframe, source : str
        Metadata forwarded to the result dict unchanged.
    htf_candles : list of dicts or None
        Higher-timeframe OHLCV rows (same schema) sorted ascending.
        When None, HTF enrichment is skipped and HTF-dependent fields default
        to neutral values.

    Returns
    -------
    dict
        Full result dict (see module docstring for field catalogue).
    None
        Returned when data is insufficient, no LTF zones form, or no event
        fires on the last bar.

    Backward compatibility
    ----------------------
    All fields from the pre-institutional pipeline are preserved:
    zones_detected, event_type, zone_center, zone_touches, vol_spike,
    vol_ratio, trend, ema_confirms, direction, signal_valid, confidence,
    stop_price, target_price, r_multiple, distance_pct.
    """
    if len(candles) < _MIN_BARS:
        return None

    # ── Step 1: indicator features ───────────────────────────────────────────
    rows  = compute_features(candles)
    curr  = rows[-1]
    atr   = curr.get("atr_14")
    close = curr.get("close")

    if not (atr and math.isfinite(atr) and atr > 0):
        return None
    if not (close and math.isfinite(close) and close > 0):
        return None

    # ── Step 2: build DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame(candles)
    df["ts"] = pd.to_datetime(df["ts"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    # ── Step 3: data quality ─────────────────────────────────────────────────
    data_quality = _assess_data_quality(df, timeframe, source)
    volume_ok    = data_quality["volume_ok"]

    # ── Step 4: pivot detection (adaptive order based on current volatility) ──
    pivot_order = adaptive_pivot_order(atr, close)
    df = find_pivots(df, order=pivot_order)

    # ── Step 5: volume signals ───────────────────────────────────────────────
    df = add_volume_signals(df)

    # ── Step 6: EMA trend filter ─────────────────────────────────────────────
    df = add_trend_filter(df)

    # ── Step 7: LTF zone clustering ──────────────────────────────────────────
    zone_tol  = atr * _ZONE_ATR_MULT
    ltf_zones = build_zones_from_pivots(df, zone_tol=zone_tol, min_touches=_MIN_TOUCHES, atr=atr)
    if not ltf_zones:
        return None
    ltf_zones = invalidate_broken_zones(ltf_zones, df, atr)
    ltf_zones = [z for z in ltf_zones if z.is_valid]
    if not ltf_zones:
        return None

    # ── Step 7.5: merge PDH/PDL zones ────────────────────────────────────────
    pdh_pdl = get_daily_levels(df, zone_tol)
    if pdh_pdl:
        ltf_zones = ltf_zones + pdh_pdl

    # ── Step 8: HTF zone clustering + bias ───────────────────────────────────
    htf_zones: List[Zone] = []
    htf_bias  = "neutral"
    htf_atr   = atr        # fallback: use LTF ATR for MTF tol
    alignment: Dict = {"linked": [], "unlinked_ltf": list(ltf_zones), "htf_only": []}
    near_htf  = False

    if htf_candles and len(htf_candles) >= _MIN_HTF_BARS:
        htf_df = pd.DataFrame(htf_candles)
        htf_df["ts"] = pd.to_datetime(htf_df["ts"])
        for col in ("open", "high", "low", "close"):
            htf_df[col] = htf_df[col].astype(float)
        if "volume" in htf_df.columns:
            htf_df["volume"] = pd.to_numeric(htf_df["volume"], errors="coerce")
        else:
            htf_df["volume"] = np.nan

        htf_rows = compute_features(htf_candles)
        htf_curr = htf_rows[-1]
        _raw_htf_atr = htf_curr.get("atr_14")
        if _raw_htf_atr and math.isfinite(_raw_htf_atr) and _raw_htf_atr > 0:
            htf_atr = _raw_htf_atr

        htf_pivot_order = adaptive_pivot_order(htf_atr, float(htf_df["close"].iloc[-1]))
        htf_df   = find_pivots(htf_df, order=htf_pivot_order)
        htf_df   = add_trend_filter(htf_df)
        htf_tol  = htf_atr * _ZONE_ATR_MULT
        htf_zones = build_zones_from_pivots(htf_df, zone_tol=htf_tol, min_touches=_MIN_TOUCHES, atr=htf_atr)
        htf_zones = invalidate_broken_zones(htf_zones, htf_df, htf_atr)
        htf_zones = [z for z in htf_zones if z.is_valid]
        htf_bias  = get_htf_bias(htf_df)

        if htf_zones:
            mtf_tol   = max(htf_tol, zone_tol)
            alignment = align_zones(htf_zones, ltf_zones, tol=mtf_tol)

    # ── Step 9: VWAP (volume-gated) ──────────────────────────────────────────
    vwap_session           = None
    vwap_anchored          = None
    vwap_session_dist_pct  = None
    vwap_anchored_dist_pct = None

    if volume_ok:
        df = add_session_vwap(df)
        last_vwap_session = float(df["vwap_session"].iloc[-1])
        if math.isfinite(last_vwap_session) and close > 0:
            vwap_session          = round(last_vwap_session, 4)
            vwap_session_dist_pct = round(abs(close - last_vwap_session) / close * 100, 4)

        anchor_idx = get_last_major_pivot_index(
            df, kind="low", lookback_pivots=5, prominence_horizon=20
        )
        if anchor_idx is None:
            anchor_idx = get_last_major_pivot_index(
                df, kind="high", lookback_pivots=5, prominence_horizon=20
            )
        if anchor_idx is not None:
            df = add_anchored_vwap(df, anchor_idx=anchor_idx)
            last_avwap = float(df["vwap_anchored"].iloc[-1])
            if math.isfinite(last_avwap) and close > 0:
                vwap_anchored          = round(last_avwap, 4)
                vwap_anchored_dist_pct = round(abs(close - last_avwap) / close * 100, 4)

    # ── Step 10: pre-select nearby zones (performance cap) ───────────────────
    nearby = _select_nearby(ltf_zones, close, _MAX_NEARBY)

    # ── Step 11: event detection — sweeps first, then structural ─────────────
    event: Optional[Dict] = None

    # Gap fades run first — a gap-open reversal is more specific than a generic
    # bounce/sweep and should not be overridden by zone-based signals.
    gap = detect_gap(df, atr)
    if gap is not None and gap["type"] in ("gap_fade_long", "gap_fade_short"):
        event = gap

    if event is None:
        event = detect_ath_breakout(df, zone_tol)

    if event is None:
        sweep = detect_liquidity_sweeps(df, nearby, zone_tol, lookback_bars=_SWEEP_LOOKBACK)
        if sweep is not None:
            event = sweep
        else:
            struct = detect_breakout_or_bounce(df, nearby, zone_tol)
            if struct is not None:
                event = struct

    # Extended detectors — run only when no primary event found yet.
    if event is None:
        event = detect_fvg(df, zone_tol, atr)
    if event is None:
        event = detect_vwap_reclaim(df, atr)
    if event is None:
        event = detect_bos(df, zone_tol)
    if event is None:
        event = detect_orb(df, atr, zone_tol)
    if event is None:
        event = detect_outside_bar(df)
    if event is None:
        event = detect_inside_bar(df)
    # Gap continuations are lowest priority (similar confidence to breakouts).
    if event is None and gap is not None:
        event = gap

    if event is None:
        return None

    # ── Step 12: gate checks ─────────────────────────────────────────────────
    last      = df.iloc[-1]
    vol_spike = bool(last["vol_spike"])
    vol_sma   = float(last["vol_sma"]) if not pd.isna(last["vol_sma"]) else None
    vol_ratio = (
        round(float(last["volume"]) / vol_sma, 2)
        if vol_sma and vol_sma > 0 and not pd.isna(last["volume"])
        else None
    )

    trend_bull = bool(last["trend_bull"])
    trend_bear = bool(last["trend_bear"])
    trend      = "bull" if trend_bull else "bear" if trend_bear else "neutral"

    direction    = _DIRECTION.get(event["type"])
    ema_confirms = bool(
        (direction == "long"  and trend_bull) or
        (direction == "short" and trend_bear)
    )

    # RSI divergence — confluence gate only (does not block the signal).
    rsi_div = detect_rsi_divergence(df)
    rsi_divergence_aligned = bool(
        (rsi_div == "bullish" and direction == "long") or
        (rsi_div == "bearish" and direction == "short")
    )

    # HTF filter — always True when no HTF data (graceful degradation).
    ltf_trend_aligned = ema_confirms
    htf_trend_aligned: Optional[bool] = None
    mtf_aligned = True

    if htf_zones:
        near_htf    = _near_any_htf_zone(float(event["zone_center"]), alignment, zone_tol)
        htf_aligned = (
            (direction == "long"  and htf_bias == "bullish") or
            (direction == "short" and htf_bias == "bearish")
        )
        htf_trend_aligned = htf_aligned
        # MTF filter: signal must agree with HTF bias OR be a sweep at HTF level.
        _fake_signal = {"type": event["type"], "zone_center": event["zone_center"]}
        mtf_aligned = filter_signal_by_htf(_fake_signal, htf_bias, alignment, zone_tol)
    else:
        near_htf = False

    # Legacy gate (backward compat): requires vol_spike + ema_confirms.
    signal_valid = bool(event["type"] and vol_spike and ema_confirms)

    # Richer status: considers volume availability and MTF alignment.
    if not volume_ok:
        signal_status = "watchlist"   # can't confirm without volume
    elif vol_spike and ema_confirms and mtf_aligned:
        signal_status = "active"
    else:
        signal_status = "watchlist"

    # ── Step 13: confluence scoring ──────────────────────────────────────────
    triggering_zone = _find_zone_by_center(nearby, event["zone_center"])

    confluence_result = score_setup(
        event                  = event,
        zone                   = triggering_zone,
        atr14                  = atr,
        vol_ratio              = vol_ratio,
        ltf_trend_aligned      = ltf_trend_aligned,
        htf_trend_aligned      = htf_trend_aligned,
        vwap_session_dist_pct  = vwap_session_dist_pct,
        vwap_anchored_dist_pct = vwap_anchored_dist_pct,
        htf_bias_aligned       = (htf_trend_aligned is True),
        near_htf_zone          = near_htf,
        ref_ts                 = df["ts"].iloc[-1],
        rsi_divergence_aligned = rsi_divergence_aligned,
    )
    confluence_score   = confluence_result["score"]
    confluence_reasons = confluence_result["reasons"]

    # ── Step 13b: ML probability score ───────────────────────────────────────
    # Returns None when model hasn't been trained yet — pipeline continues
    # using confluence_score as the primary ranking signal until then.
    _ml_signal = {
        "event_type":            event["type"],
        "direction":             direction,
        "signal_status":         signal_status,
        "zone_touches":          event["touches"],
        "zones_ltf_count":       len(ltf_zones),
        "zones_htf_count":       len(htf_zones),
        "distance_pct":          round(abs(close - event["zone_center"]) / close * 100, 4) if event.get("zone_center") else 0.0,
        "vol_ratio":             vol_ratio,
        "vol_spike":             vol_spike,
        "ema_confirms":          ema_confirms,
        "atr_14":                atr,
        "trend":                 trend,
        "htf_bias":              htf_bias,
        "mtf_aligned":           mtf_aligned,
        "near_htf_zone":         near_htf,
        "vwap_session_dist_pct":  vwap_session_dist_pct,
        "vwap_anchored_dist_pct": vwap_anchored_dist_pct,
        "confluence_score":      confluence_score,
        "confluence_reasons":    confluence_reasons,
    }
    ml_probability = get_predictor().score(_ml_signal)

    # Legacy confidence score (preserved for backward compat with scan.py).
    zone_touches = event["touches"]
    confidence   = _confidence(zone_touches, vol_spike, ema_confirms)

    # ── Step 14: position sizing ─────────────────────────────────────────────
    if direction == "long":
        stop_price   = round(close - ATR_STOP_MULT   * atr, 4)
        target_price = round(close + ATR_TARGET_MULT * atr, 4)
    else:
        stop_price   = round(close + ATR_STOP_MULT   * atr, 4)
        target_price = round(close - ATR_TARGET_MULT * atr, 4)

    zone_center  = event["zone_center"]
    distance_pct = (
        round(abs(close - zone_center) / close * 100, 4) if zone_center else None
    )

    return {
        # ── Backward-compatible fields ────────────────────────────────────
        # Scan metrics
        "zones_detected":  len(ltf_zones),
        # Metadata
        "symbol":          symbol,
        "timeframe":       timeframe,
        "source":          source,
        "ts":              curr["ts"],
        "close":           close,
        "ema_20":          curr.get("ema_20"),
        "ema_50":          curr.get("ema_50"),
        "rsi_14":          curr.get("rsi_14"),
        "atr_14":          atr,
        # Structure event
        "event_type":      event["type"],
        "zone_center":     zone_center,
        "zone_touches":    zone_touches,
        # Volume
        "vol_spike":       vol_spike,
        "vol_ratio":       vol_ratio,
        # Trend
        "trend":           trend,
        "ema_confirms":    ema_confirms,
        # Signal validity (legacy gate)
        "direction":       direction,
        "signal_valid":    signal_valid,
        "confidence":      confidence,
        # Position sizing
        "stop_price":      stop_price,
        "target_price":    target_price,
        "r_multiple":      R_MULTIPLE,
        # Proximity
        "distance_pct":    distance_pct,

        # ── New institutional-lite fields ─────────────────────────────────
        # Zone counts
        "zones_ltf_count": len(ltf_zones),
        "zones_htf_count": len(htf_zones),
        # HTF
        "htf_bias":        htf_bias,
        "near_htf_zone":   near_htf,
        "mtf_aligned":     mtf_aligned,
        # VWAP
        "vwap_session":           vwap_session,
        "vwap_anchored":          vwap_anchored,
        "vwap_session_dist_pct":  vwap_session_dist_pct,
        "vwap_anchored_dist_pct": vwap_anchored_dist_pct,
        # Zones (serialized for JSON consumers)
        "zones_ltf": _serialize_zones(ltf_zones),
        "zones_htf": _serialize_zones(htf_zones),
        # Confluence scoring
        "confluence_score":   confluence_score,
        "confluence_reasons": confluence_reasons,
        # Signal status
        "signal_status":   signal_status,
        # ML probability (None until model is trained)
        "ml_probability":  ml_probability,
        # Data quality
        "data_quality":    data_quality,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_nearby(zones: List[Zone], price: float, n: int) -> List[Zone]:
    """Return the *n* zones with centers closest to *price*.

    O(N) argpartition — safe even with many zones.
    """
    if len(zones) <= n:
        return zones
    centers = np.array([z.center for z in zones], dtype=np.float64)
    dists   = np.abs(centers - price)
    idx     = np.argpartition(dists, n)[:n]
    return [zones[i] for i in idx]


def _find_zone_by_center(zones: List[Zone], center: float) -> Zone:
    """Return the Zone whose center matches *center*.  Falls back to zones[0]."""
    for z in zones:
        if abs(z.center - center) < 1e-6:
            return z
    return zones[0]


def _serialize_zones(zones: List[Zone]) -> List[Dict[str, Any]]:
    """Convert Zone dataclass objects to JSON-serializable dicts."""
    return [
        {
            "kind":       z.kind,
            "price_low":  z.low,
            "price_high": z.high,
            "center":     z.center,
            "touches":    z.touches,
            "first_ts":   z.first_ts.isoformat(),
            "last_ts":    z.last_ts.isoformat(),
        }
        for z in zones
    ]


def _assess_data_quality(
    df: pd.DataFrame,
    timeframe: str,
    source: Optional[str],
) -> Dict[str, Any]:
    """
    Produce a lightweight data-quality summary for a symbol.

    volume_ok
        True when ≥ 80 % of bars have a positive non-NaN volume value.
        When False, VWAP and volume-spike gates are skipped and signal_status
        is forced to "watchlist" regardless of other signals.

    gaps_detected
        True when the observed bar count is < 80 % of the bars that would
        be expected given the date range (daily / weekly timeframes only;
        intraday uses a consecutive-delta heuristic).

    provider
        Forwarded from the `source` argument.  Set to "mixed" when the
        source string contains commas (multi-provider ingestion).
    """
    n = len(df)

    # --- Volume ---
    if "volume" in df.columns:
        vol_series  = pd.to_numeric(df["volume"], errors="coerce")
        valid_count = int((vol_series > 0).sum())
        volume_ok   = valid_count >= int(n * 0.80)
    else:
        volume_ok = False

    # --- Gaps ---
    gaps_detected = _detect_gaps(df, timeframe)

    # --- Provider ---
    if source and "," in source:
        provider = "mixed"
    else:
        provider = source or "unknown"

    last_ts = df["ts"].iloc[-1] if not df.empty else None

    return {
        "provider":       provider,
        "volume_ok":      volume_ok,
        "gaps_detected":  gaps_detected,
        "bars_count":     n,
        "last_ts":        last_ts,
    }


def _detect_gaps(df: pd.DataFrame, timeframe: str) -> bool:
    """
    Heuristic gap detector.

    For daily / weekly timeframes: compares observed bar count against the
    expected trading-day count derived from the calendar span.  Missing > 20 %
    of bars signals a problematic data gap.

    For intraday: uses consecutive timestamp deltas.  A delta > 4× the median
    intra-session delta indicates an unexpected gap (overnight is handled by
    using the 20th-percentile delta as the baseline to exclude normal market-
    hours-only gaps).
    """
    if len(df) < 4:
        return False

    ts: pd.Series = df["ts"]

    if timeframe in ("1d", "1w"):
        span_days = (ts.iloc[-1] - ts.iloc[0]).days
        if span_days <= 0:
            return False
        if timeframe == "1d":
            expected = span_days * 5 / 7   # ~trading days per calendar day
        else:
            expected = span_days / 7
        return len(df) < expected * 0.80

    # Intraday: percentile-based gap detection.
    deltas_sec = ts.diff().dt.total_seconds().dropna()
    if deltas_sec.empty:
        return False
    # Use 20th percentile as the "normal" intra-session interval to avoid
    # overnight gaps inflating the baseline.
    pct20 = float(np.percentile(deltas_sec, 20))
    if pct20 <= 0:
        return False
    return bool((deltas_sec > 4 * pct20).any())


def _confidence(touches: int, vol_spike: bool, ema_confirms: bool) -> float:
    """Legacy composite confidence score in [0.0, 1.0] (preserved for
    backward compatibility with scan.py and existing API consumers).

    Zone strength  : 0.15 per touch, capped at 0.60
    Volume spike   : +0.20
    EMA alignment  : +0.20
    """
    zone_score = min(0.60, touches * 0.15)
    vol_score  = 0.20 if vol_spike   else 0.0
    ema_score  = 0.20 if ema_confirms else 0.0
    return round(min(1.0, zone_score + vol_score + ema_score), 4)

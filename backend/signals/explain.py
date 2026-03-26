"""backend/signals/explain.py

Plain-English explanation formatter for trading signals.

Converts the structured pipeline output dict into human-readable text that
a non-technical user can understand at a glance.

Example output
--------------
  "AAPL is testing a 4-touch support zone near $182.50. Volume is running
   2.1× above average with the trend confirming bullish bias. Higher
   timeframe zones align — model gives this a 67% chance of bouncing."
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Event type → plain English description
# ---------------------------------------------------------------------------

_EVENT_DESCRIPTIONS: Dict[str, str] = {
    "breakout_up":              "breaking above a resistance zone",
    "breakdown_down":           "breaking below a support zone",
    "bounce_up":                "bouncing off a support zone",
    "reject_down":              "rejecting at a resistance zone",
    "sweep_up":                 "sweeping liquidity above a key level",
    "sweep_down":               "sweeping liquidity below a key level",
    "bos_up":                   "showing a bullish break of structure",
    "bos_down":                 "showing a bearish break of structure",
    "choch_up":                 "showing a bullish change of character",
    "choch_down":               "showing a bearish change of character",
    "fvg_long":                 "filling a bullish fair value gap",
    "fvg_short":                "filling a bearish fair value gap",
    "vwap_reclaim_long":        "reclaiming VWAP from below",
    "vwap_reclaim_short":       "losing VWAP from above",
    "orb_long":                 "breaking above the opening range",
    "orb_short":                "breaking below the opening range",
    "gap_fade_long":            "fading a downside gap open",
    "gap_fade_short":           "fading an upside gap open",
    "gap_go_long":              "continuing above a gap open",
    "gap_go_short":             "continuing below a gap open",
    "ath_breakout_long":        "breaking into all-time high territory",
    "inside_bar_long":          "breaking out of an inside bar to the upside",
    "inside_bar_short":         "breaking out of an inside bar to the downside",
    "double_inside_bar_long":   "breaking out of a double inside bar consolidation",
    "double_inside_bar_short":  "breaking down from a double inside bar consolidation",
    "outside_bar_long":         "showing a bullish engulfing bar",
    "outside_bar_short":        "showing a bearish engulfing bar",
}

_DIRECTION_LABEL: Dict[str, str] = {
    "long":  "upside",
    "short": "downside",
}

_TREND_LABEL: Dict[str, str] = {
    "bull":    "bullish",
    "bear":    "bearish",
    "neutral": "neutral",
}

_HTF_BIAS_LABEL: Dict[str, str] = {
    "bullish": "bullish",
    "bearish": "bearish",
    "neutral": "neutral",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_signal(result: Dict[str, Any]) -> str:
    """
    Generate a plain-English explanation of a pipeline result.

    Parameters
    ----------
    result : dict
        Output from run_structure_pipeline() — must contain at minimum
        symbol, event_type, direction, and close.

    Returns
    -------
    str
        2-3 sentence human-readable explanation.
    """
    symbol      = result.get("symbol", "This ticker")
    event_type  = result.get("event_type", "")
    direction   = result.get("direction", "long")
    close       = result.get("close")
    zone_center = result.get("zone_center")
    zone_touches = int(result.get("zone_touches") or 0)
    vol_ratio   = result.get("vol_ratio")
    vol_spike   = result.get("vol_spike", False)
    ema_confirms = result.get("ema_confirms", False)
    trend       = result.get("trend", "neutral")
    htf_bias    = result.get("htf_bias", "neutral") or "neutral"
    near_htf    = result.get("near_htf_zone", False)
    mtf_aligned = result.get("mtf_aligned", False)
    ml_prob     = result.get("ml_probability")
    conf_score  = result.get("confluence_score", 0)
    signal_status = result.get("signal_status", "watchlist")
    vwap_session  = result.get("vwap_session")
    vwap_dist     = result.get("vwap_session_dist_pct")

    parts = []

    # ── Sentence 1: What is happening ────────────────────────────────────
    event_desc = _EVENT_DESCRIPTIONS.get(event_type, f"triggering a {event_type} signal")
    zone_str   = f" near ${zone_center:,.2f}" if zone_center else ""
    touch_str  = (
        f" ({zone_touches}-touch zone)" if zone_touches >= 3
        else f" ({zone_touches}-touch zone)" if zone_touches >= 2
        else ""
    )
    parts.append(f"{symbol} is {event_desc}{zone_str}{touch_str}.")

    # ── Sentence 2: What's confirming it ─────────────────────────────────
    confirmations = []

    if vol_spike and vol_ratio is not None:
        confirmations.append(f"volume is {vol_ratio:.1f}× above average")
    elif vol_ratio is not None and vol_ratio >= 1.2:
        confirmations.append(f"volume is elevated at {vol_ratio:.1f}×")
    else:
        confirmations.append("volume is below average — lower conviction")

    trend_label = _TREND_LABEL.get(trend, "neutral")
    if ema_confirms:
        confirmations.append(f"trend is confirming {trend_label} bias")
    else:
        confirmations.append(f"trend is {trend_label} but not yet confirming")

    if vwap_session is not None and vwap_dist is not None:
        if vwap_dist <= 0.3:
            confirmations.append(f"price is at VWAP (${vwap_session:,.2f})")
        elif vwap_dist <= 1.0:
            confirmations.append(f"price is near VWAP (${vwap_session:,.2f})")

    if confirmations:
        parts.append(", ".join(confirmations[:3]).capitalize() + ".")

    # ── Sentence 3: HTF context + probability ────────────────────────────
    context_parts = []

    htf_label = _HTF_BIAS_LABEL.get(htf_bias, "neutral")
    if near_htf and mtf_aligned:
        context_parts.append(f"higher timeframe is {htf_label} and price is at an HTF zone")
    elif near_htf:
        context_parts.append("price is at a higher timeframe zone")
    elif mtf_aligned:
        context_parts.append(f"higher timeframe trend is {htf_label}")

    # Probability statement
    if ml_prob is not None:
        pct = round(ml_prob * 100)
        direction_label = _DIRECTION_LABEL.get(direction, direction)
        prob_str = f"model gives this a {pct}% probability of {direction_label} follow-through"
        context_parts.append(prob_str)
    elif conf_score >= 60:
        direction_label = _DIRECTION_LABEL.get(direction, direction)
        context_parts.append(
            f"confluence score is {conf_score}/100 — moderately favorable for {direction_label}"
        )

    if context_parts:
        parts.append(", ".join(context_parts).capitalize() + ".")

    # ── Watchlist caveat ──────────────────────────────────────────────────
    if signal_status == "watchlist":
        parts.append("Note: volume or trend confirmation is missing — treat as watchlist only.")

    return " ".join(parts)


def format_alert(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a pipeline result into a clean alert payload for the API and
    notification system.

    Returns a dict with all fields needed by the frontend and notification
    dispatcher, including the plain-English explanation.
    """
    ml_prob    = result.get("ml_probability")
    conf_score = result.get("confluence_score", 0)

    # Probability: prefer ML, fall back to scaled confluence_score
    if ml_prob is not None:
        probability        = round(ml_prob, 4)
        probability_source = "ml_model"
    else:
        probability        = round(conf_score / 100, 4)
        probability_source = "confluence_score"

    direction = result.get("direction", "long")
    direction_label = _DIRECTION_LABEL.get(direction, direction)

    return {
        "symbol":             result.get("symbol"),
        "timeframe":          result.get("timeframe"),
        "direction":          direction,
        "direction_label":    direction_label,
        "event_type":         result.get("event_type"),
        "signal_status":      result.get("signal_status"),
        "probability":        probability,
        "probability_source": probability_source,
        "probability_pct":    f"{round(probability * 100)}%",
        "confluence_score":   conf_score,
        "explanation":        explain_signal(result),
        "key_factors":        _key_factors(result),
        "price": {
            "current":  result.get("close"),
            "entry":    result.get("close"),
            "stop":     result.get("stop_price"),
            "target":   result.get("target_price"),
            "r_multiple": result.get("r_multiple"),
        },
        "zone": {
            "center":  result.get("zone_center"),
            "touches": result.get("zone_touches"),
        },
        "ts": result.get("ts"),
    }


def format_no_signal(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Return a structured 'no signal' response for the analyze endpoint."""
    return {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "signal_status": "no_signal",
        "probability":   None,
        "explanation":   f"No active setup detected for {symbol} on the {timeframe} timeframe right now.",
        "key_factors":   [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_factors(result: Dict[str, Any]) -> list:
    """Extract the 3-5 most relevant factors as short bullet strings."""
    factors = []

    zone_touches = int(result.get("zone_touches") or 0)
    if zone_touches >= 2:
        factors.append(f"{zone_touches}-touch zone")

    vol_ratio = result.get("vol_ratio")
    vol_spike = result.get("vol_spike", False)
    if vol_spike and vol_ratio:
        factors.append(f"Volume {vol_ratio:.1f}× average")
    elif not vol_spike:
        factors.append("No volume confirmation")

    if result.get("ema_confirms"):
        factors.append("EMA trend confirmed")

    if result.get("near_htf_zone"):
        factors.append("At HTF zone")

    if result.get("mtf_aligned"):
        factors.append("HTF trend aligned")

    htf_bias = result.get("htf_bias", "neutral") or "neutral"
    if htf_bias in ("bullish", "bearish") and not result.get("mtf_aligned"):
        factors.append(f"HTF bias: {htf_bias}")

    if result.get("confluence_score", 0) >= 70:
        factors.append(f"High confluence ({result['confluence_score']}/100)")

    return factors[:5]

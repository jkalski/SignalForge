"""
Breakout and bounce detection against support / resistance zones.

Each event is evaluated on the most recent closed bar in `df` against every
zone supplied.  The strongest qualifying event is returned (or None).

Event types
-----------
breakout_up   — close convincingly above resistance (close > zone.high + zone_tol)
breakdown_down — close convincingly below support   (close < zone.low  - zone_tol)
bounce_up      — wick into support, body closes above (low  <= zone.high, close > zone.high)
reject_down    — wick into resistance, body closes below (high >= zone.low,  close < zone.low)

Priority
--------
When multiple zones fire on the same bar, the result is selected by:
  1. breakout / breakdown before bounce / rejection  (more definitive price action)
  2. highest zone touch count within the same type   (stronger structural level)

The zones list from build_zones_from_pivots() is already sorted by touches
descending, so the first matching zone of a given type is always the strongest.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from backend.indicators.zones import Zone


# Lower number = higher priority in tiebreaking.
_TYPE_PRIORITY: Dict[str, int] = {
    "breakout_up":    0,
    "breakdown_down": 1,
    "bounce_up":      2,
    "reject_down":    3,
}


def detect_breakout_or_bounce(
    df: pd.DataFrame,
    zones: List[Zone],
    zone_tol: float,
) -> Optional[Dict]:
    """
    Classify the most recent bar's price action relative to known zones.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.  Must contain columns:
        ts, high, low, close.  Only the last row is evaluated.
    zones : List[Zone]
        Output of ``build_zones_from_pivots()``, sorted by touches desc.
    zone_tol : float
        Tolerance used for breakout confirmation.  Recommended: ATR_14 * 0.5.
        Same value used when building zones ensures consistency.

    Returns
    -------
    dict or None
        The highest-priority event found, with keys:
            type       : str   — event type (see module docstring)
            zone_center: float — midpoint of the triggering zone
            touches    : int   — how many pivots formed the zone
            timestamp  : any   — ts of the bar that triggered the event
        None if no event qualifies on the last bar.

    Raises
    ------
    ValueError
        If required columns are missing or zone_tol <= 0.
    """
    _validate(df, zone_tol)

    if not zones:
        return None

    bar = df.iloc[-1]
    close = float(bar["close"])
    high  = float(bar["high"])
    low   = float(bar["low"])
    ts    = bar["ts"]

    candidates: List[Dict] = []

    for zone in zones:
        event_type: Optional[str] = None

        if zone.kind == "resistance":
            if close > zone.high + zone_tol:
                # Body closes cleanly above the entire resistance zone.
                event_type = "breakout_up"
            elif high >= zone.low and close < zone.low:
                # Wick enters resistance from below; body rejected back under it.
                event_type = "reject_down"

        elif zone.kind == "support":
            if close < zone.low - zone_tol:
                # Body closes cleanly below the entire support zone.
                event_type = "breakdown_down"
            elif low <= zone.high and close > zone.high:
                # Wick dips into support; body closes back above it.
                event_type = "bounce_up"

        if event_type is not None:
            candidates.append(
                {
                    "type":        event_type,
                    "zone_center": zone.center,
                    "touches":     zone.touches,
                    "timestamp":   ts,
                    # Internal sort keys — stripped before returning.
                    "_priority":   _TYPE_PRIORITY[event_type],
                    "_touches":    zone.touches,
                }
            )

    if not candidates:
        return None

    # Prefer definitive breakouts over wick-based events; within same type,
    # prefer the zone with the most structural touches.
    candidates.sort(key=lambda c: (c["_priority"], -c["_touches"]))

    best = candidates[0]
    return {
        "type":        best["type"],
        "zone_center": best["zone_center"],
        "touches":     best["touches"],
        "timestamp":   best["timestamp"],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, zone_tol: float) -> None:
    required = {"ts", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if zone_tol <= 0:
        raise ValueError(f"zone_tol must be > 0, got {zone_tol}")

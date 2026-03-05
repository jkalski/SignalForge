"""
Liquidity sweep (stop-hunt) detection against support / resistance zones.

A liquidity sweep occurs when price briefly trades beyond a known zone
boundary — triggering resting stop orders — then closes back inside,
signalling a potential reversal.

Sweep up  (buy-side liquidity grab above resistance highs):
    Within the lookback window at least one bar's high exceeds the zone's
    upper boundary (zone.high + zone_tol).  The confirming (last) bar then
    closes back below that boundary (close < zone.high).

Sweep down (sell-side liquidity grab below support lows):
    Within the lookback window at least one bar's low pierces below the
    zone's lower boundary (zone.low - zone_tol).  The confirming bar then
    closes back above that boundary (close > zone.low).

Performance
-----------
Only the ``_MAX_NEARBY_ZONES`` zones nearest to the current close are
evaluated; distant zones are skipped with a single O(N) numpy partition.
All wick-pierce checks use pre-computed scalar window extremes — no
per-bar Python loops.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.indicators.zones import Zone

_MAX_NEARBY_ZONES = 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    zones: List[Zone],
    zone_tol: float,
    lookback_bars: int = 2,
) -> Optional[Dict]:
    """
    Check the most recent ``lookback_bars`` candles for a completed
    liquidity sweep against any nearby support / resistance zone.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.  Required columns:
        ``ts``, ``high``, ``low``, ``close``.  The ``volume`` column is
        ignored so the function is robust to missing volume data.
        Must have at least ``lookback_bars`` rows.
    zones : List[Zone]
        Output of ``build_zones_from_pivots()``, sorted by touches desc.
        Only the 6 zones nearest to the current price are examined.
    zone_tol : float
        Minimum wick distance beyond the zone boundary to count as a
        pierce.  Recommended: ``ATR_14 * 0.5`` (same value used when
        building zones).
    lookback_bars : int, default 2
        How many trailing bars to search for the wick-pierce event.
        The close-reclaim test always uses only the last bar.
        ``lookback_bars=1`` requires the sweep to complete within a
        single candle.

    Returns
    -------
    dict or None
        Strongest qualifying sweep, with keys::

            type        str   — "sweep_up" | "sweep_down"
            zone_kind   str   — "resistance" | "support"
            zone_center float — midpoint of the triggering zone
            touches     int   — pivot count that formed the zone
            ts          any   — ``ts`` value of the confirming (last) bar
            close       float — close price of the confirming bar

        ``None`` when no sweep qualifies.

    Raises
    ------
    ValueError
        Missing columns, empty DataFrame, ``zone_tol <= 0``, or
        ``lookback_bars < 1``.
    """
    _validate(df, zone_tol, lookback_bars)

    if not zones:
        return None

    last_bar = df.iloc[-1]
    last_close = float(last_bar["close"])
    last_ts = last_bar["ts"]

    # ---- 1. Restrict to nearest zones (O(N) argpartition) ---------------
    nearby = _nearest_zones(zones, last_close, _MAX_NEARBY_ZONES)
    if not nearby:
        return None

    # ---- 2. Window extremes (single vectorized pass) --------------------
    window = df.iloc[-lookback_bars:]
    max_high = float(window["high"].to_numpy(dtype=np.float64).max())
    min_low = float(window["low"].to_numpy(dtype=np.float64).min())

    # ---- 3. Evaluate each nearby zone -----------------------------------
    candidates: List[Dict] = []

    for zone in nearby:
        if zone.kind == "resistance":
            pierce_threshold = zone.high + zone_tol
            if max_high > pierce_threshold and last_close < zone.high:
                # Wick ran above resistance, close confirmed back below.
                candidates.append(
                    {
                        "type": "sweep_up",
                        "zone_kind": "resistance",
                        "zone_center": zone.center,
                        "touches": zone.touches,
                        "ts": last_ts,
                        "close": last_close,
                        "_depth": max_high - zone.high,
                        "_touches": zone.touches,
                    }
                )

        elif zone.kind == "support":
            pierce_threshold = zone.low - zone_tol
            if min_low < pierce_threshold and last_close > zone.low:
                # Wick ran below support, close confirmed back above.
                candidates.append(
                    {
                        "type": "sweep_down",
                        "zone_kind": "support",
                        "zone_center": zone.center,
                        "touches": zone.touches,
                        "ts": last_ts,
                        "close": last_close,
                        "_depth": zone.low - min_low,
                        "_touches": zone.touches,
                    }
                )

    if not candidates:
        return None

    # ---- 4. Pick the best candidate -------------------------------------
    # Deeper sweep first (larger wick beyond zone = more stops run).
    # Tiebreak: zone with more touches is more structurally significant.
    candidates.sort(key=lambda c: (-c["_depth"], -c["_touches"]))
    best = candidates[0]

    return {
        "type": best["type"],
        "zone_kind": best["zone_kind"],
        "zone_center": best["zone_center"],
        "touches": best["touches"],
        "ts": best["ts"],
        "close": best["close"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_zones(zones: List[Zone], price: float, n: int) -> List[Zone]:
    """Return the *n* zones whose center is closest to *price*.

    Uses ``np.argpartition`` for O(N) selection without a full sort.
    """
    if len(zones) <= n:
        return zones

    centers = np.array([z.center for z in zones], dtype=np.float64)
    dists = np.abs(centers - price)
    # argpartition guarantees the first n indices are the n smallest
    # distances (not necessarily sorted among themselves).
    idx = np.argpartition(dists, n)[:n]
    return [zones[i] for i in idx]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate(df: pd.DataFrame, zone_tol: float, lookback_bars: int) -> None:
    required = {"ts", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if zone_tol <= 0:
        raise ValueError(f"zone_tol must be > 0, got {zone_tol!r}")
    if lookback_bars < 1:
        raise ValueError(f"lookback_bars must be >= 1, got {lookback_bars!r}")
    if len(df) < lookback_bars:
        raise ValueError(
            f"DataFrame has {len(df)} row(s) but lookback_bars={lookback_bars}; "
            "need at least that many rows."
        )

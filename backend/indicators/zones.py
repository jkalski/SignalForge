"""
Support and resistance zone detection via pivot clustering.

Algorithm
---------
1. Extract pivot highs as resistance candidates (df['high'] where pivot_high=True).
2. Extract pivot lows as support candidates  (df['low']  where pivot_low=True).
3. For each set, sort pivots by price then sweep once with np.diff to identify
   gaps larger than `zone_tol`.  np.cumsum over the gap mask assigns a cluster
   ID to every pivot in O(N) with no Python row loops.
4. Aggregate each cluster into a Zone: min/max price, touch count, timestamps.
5. Filter clusters with fewer than `min_touches` pivots.
6. Sort surviving zones by (touches desc, last_ts desc).
7. Enrich each zone with ML features when atr > 0 is supplied.

ML enrichment (optional, enabled when atr > 0)
------------------------------------------------
zone_age_bars     : bars elapsed since the last bar that overlapped the zone.
zone_reaction_avg : mean bounce size across all zone touches, in ATR multiples.
                    A "successful" touch produces ≥ 0.5 ATR movement in the
                    expected direction within reaction_window forward bars.
failed_tests      : count of touches that produced < 0.5 ATR reaction.
volume_at_zone    : mean volume across all bars that overlap the zone.
strength_score    : composite 0.0–1.0 = 0.35*touches + 0.35*reaction
                    + 0.20*age_decay + 0.10*reliability

Zone invalidation
-----------------
Call invalidate_broken_zones() after building zones.  A zone is marked
is_valid=False when price closes beyond it by more than conviction_mult * ATR.

Complexity
----------
Clustering  : O(N log N) via np.argsort
Enrichment  : O(K * N)  where K = zone count (typically < 30) — acceptable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REACTION_WINDOW = 5      # forward bars used to measure zone reaction
_AGE_DECAY_BARS  = 100    # characteristic decay length for age weighting


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """A clustered support or resistance price zone."""

    kind:     Literal["support", "resistance"]
    low:      float          # lowest pivot price in the cluster
    high:     float          # highest pivot price in the cluster
    center:   float          # midpoint of [low, high]
    touches:  int            # number of pivots merged into this zone
    first_ts: pd.Timestamp   # timestamp of the earliest contributing pivot
    last_ts:  pd.Timestamp   # timestamp of the most recent contributing pivot

    # --- ML / strength fields  (defaults → fully backward-compatible) ------
    strength_score:    float = 0.0    # composite strength 0.0–1.0
    zone_age_bars:     int   = 0      # bars since the last overlapping bar
    zone_reaction_avg: float = 0.0    # average bounce in ATR multiples
    failed_tests:      int   = 0      # touches without a meaningful reaction
    volume_at_zone:    float = 0.0    # mean volume on touching bars
    is_htf_confluence: bool  = False  # True when aligned with an HTF zone
    is_valid:          bool  = True   # False when price closed through it

    @property
    def width(self) -> float:
        """Price width of the zone."""
        return self.high - self.low


# ---------------------------------------------------------------------------
# Public API — zone building
# ---------------------------------------------------------------------------

def build_zones_from_pivots(
    df: pd.DataFrame,
    zone_tol: float,
    min_touches: int = 2,
    atr: float = 0.0,
) -> List[Zone]:
    """
    Cluster pivot highs and pivot lows into support / resistance zones.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``find_pivots()``.  Required columns:
        ts, high, low, pivot_high (bool), pivot_low (bool).
        Optional columns used for ML enrichment: close, volume.
        Rows must be in chronological order.
    zone_tol : float
        Maximum price distance between two pivots for them to be merged
        into the same zone.  Recommended: ATR_14 * 0.5.
    min_touches : int, default 2
        Zones with fewer pivot touches are discarded.
    atr : float, default 0.0
        Current ATR14.  When > 0 and finite, enables ML feature enrichment
        (zone_age_bars, zone_reaction_avg, failed_tests, volume_at_zone,
        strength_score).

    Returns
    -------
    List[Zone]
        Zones sorted by (touches desc, last_ts desc).
        Empty list if no qualifying zones exist.

    Raises
    ------
    ValueError
        If required columns are missing, zone_tol <= 0, or min_touches < 1.
    """
    _validate(df, zone_tol, min_touches)

    zones: List[Zone] = []

    if df["pivot_high"].any():
        mask       = df["pivot_high"]
        prices     = df.loc[mask, "high"].to_numpy(dtype=np.float64)
        timestamps = df.loc[mask, "ts"].to_numpy()
        zones.extend(_cluster_pivots(prices, timestamps, zone_tol, "resistance"))

    if df["pivot_low"].any():
        mask       = df["pivot_low"]
        prices     = df.loc[mask, "low"].to_numpy(dtype=np.float64)
        timestamps = df.loc[mask, "ts"].to_numpy()
        zones.extend(_cluster_pivots(prices, timestamps, zone_tol, "support"))

    zones = [z for z in zones if z.touches >= min_touches]

    # Optional ML enrichment — only when ATR is supplied and valid.
    if atr > 0 and math.isfinite(atr):
        zones = [_enrich_zone(z, df, atr) for z in zones]

    zones.sort(key=lambda z: (z.touches, z.last_ts), reverse=True)
    return zones


# ---------------------------------------------------------------------------
# Public API — zone invalidation
# ---------------------------------------------------------------------------

def invalidate_broken_zones(
    zones: List[Zone],
    df: pd.DataFrame,
    atr: float,
    conviction_mult: float = 1.0,
) -> List[Zone]:
    """
    Mark zones as invalid when price closes convincingly through them.

    A **support** zone is invalidated when any close after the zone's
    last_ts falls below ``zone.low - conviction_mult * atr``.

    A **resistance** zone is invalidated when any close after the zone's
    last_ts rises above ``zone.high + conviction_mult * atr``.

    Zones are mutated in-place (``is_valid`` set to False).  The caller
    may then filter with ``[z for z in zones if z.is_valid]``.

    Parameters
    ----------
    zones : List[Zone]
    df : pd.DataFrame
        Must contain ``ts`` and ``close`` columns.
    atr : float
        Current ATR14 for the breach threshold.
    conviction_mult : float, default 1.0
        Multiplier on ATR.  Higher values require a more decisive close-
        through before invalidation.

    Returns
    -------
    List[Zone]
        Same list (mutated in-place).
    """
    if not zones or atr <= 0 or "close" not in df.columns:
        return zones

    closes = df["close"].to_numpy(dtype=np.float64)
    ts_arr = df["ts"].to_numpy()
    breach = conviction_mult * atr

    for zone in zones:
        if not zone.is_valid:
            continue
        post_mask    = ts_arr > zone.last_ts.to_datetime64()
        post_closes  = closes[post_mask]
        if len(post_closes) == 0:
            continue
        if zone.kind == "support":
            if np.any(post_closes < zone.low - breach):
                zone.is_valid = False
        else:
            if np.any(post_closes > zone.high + breach):
                zone.is_valid = False

    return zones


# ---------------------------------------------------------------------------
# Core clustering — vectorized
# ---------------------------------------------------------------------------

def _cluster_pivots(
    prices: np.ndarray,
    timestamps: np.ndarray,
    zone_tol: float,
    kind: Literal["support", "resistance"],
) -> List[Zone]:
    """Group price levels into zones by sorting then gap-detection."""
    if len(prices) == 0:
        return []

    sort_idx       = np.argsort(prices, kind="stable")
    sorted_prices  = prices[sort_idx]
    sorted_ts      = timestamps[sort_idx]

    gaps   = np.diff(sorted_prices) > zone_tol
    starts = np.concatenate([[0], np.where(gaps)[0] + 1])
    ends   = np.concatenate([starts[1:], [len(sorted_prices)]])

    low_prices    = sorted_prices[starts]
    high_prices   = sorted_prices[ends - 1]
    center_prices = (low_prices + high_prices) * 0.5
    touches       = ends - starts

    first_ts = np.minimum.reduceat(sorted_ts, starts)
    last_ts  = np.maximum.reduceat(sorted_ts, starts)

    n_clusters = len(starts)
    return [
        Zone(
            kind=kind,
            low=float(low_prices[i]),
            high=float(high_prices[i]),
            center=float(center_prices[i]),
            touches=int(touches[i]),
            first_ts=pd.Timestamp(first_ts[i]),
            last_ts=pd.Timestamp(last_ts[i]),
        )
        for i in range(n_clusters)
    ]


# ---------------------------------------------------------------------------
# ML enrichment
# ---------------------------------------------------------------------------

def _enrich_zone(
    zone: Zone,
    df: pd.DataFrame,
    atr: float,
    reaction_window: int = _REACTION_WINDOW,
) -> Zone:
    """
    Populate ML feature fields on *zone* using the full candle DataFrame.

    Scans all bars that overlap the zone's price range [low, high] and
    measures forward reactions.  Mutates and returns the same Zone object.
    """
    n      = len(df)
    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64) if "close" in df.columns else None

    # All bars whose price range overlaps the zone.
    touch_mask = (lows <= zone.high) & (highs >= zone.low)
    touch_idxs = np.where(touch_mask)[0]

    if len(touch_idxs) == 0:
        zone.strength_score = _compute_strength(zone)
        return zone

    # Bars elapsed since the most recent touch.
    zone.zone_age_bars = int(n - 1 - int(touch_idxs[-1]))

    # Average volume on touching bars.
    if "volume" in df.columns:
        vols       = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
        touch_vols = vols[touch_mask]
        valid_vols = touch_vols[np.isfinite(touch_vols) & (touch_vols > 0)]
        zone.volume_at_zone = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 0.0

    # Reaction analysis — measure forward close movement for each touch.
    if closes is not None:
        reactions: List[float] = []
        failed = 0
        for idx in touch_idxs:
            i      = int(idx)
            fwd    = closes[i + 1 : min(i + reaction_window + 1, n)]
            if len(fwd) == 0:
                continue
            if zone.kind == "support":
                move = (float(np.max(fwd)) - zone.center) / atr
            else:
                move = (zone.center - float(np.min(fwd))) / atr

            if move >= 0.5:
                reactions.append(move)
            else:
                failed += 1

        zone.zone_reaction_avg = float(np.mean(reactions)) if reactions else 0.0
        zone.failed_tests      = failed

    zone.strength_score = _compute_strength(zone)
    return zone


def _compute_strength(zone: Zone) -> float:
    """
    Composite zone strength score in [0.0, 1.0].

    Weight budget
    -------------
    35%  touch count          — more tests = more institutional interest
    35%  reaction avg         — bigger bounces = price respects the zone
    20%  age decay            — exponential decay (half-life = AGE_DECAY_BARS)
    10%  reliability          — penalises zones with many failed tests
    """
    touch_score    = min(zone.touches / 6.0, 1.0)
    reaction_score = min(zone.zone_reaction_avg / 3.0, 1.0)
    age_decay      = math.exp(-zone.zone_age_bars / _AGE_DECAY_BARS)
    failure_mult   = max(0.0, 1.0 - zone.failed_tests * 0.15)

    raw = (
        0.35 * touch_score
        + 0.35 * reaction_score
        + 0.20 * age_decay
        + 0.10 * failure_mult
    )
    return round(min(raw, 1.0), 4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, zone_tol: float, min_touches: int) -> None:
    required = {"ts", "high", "low", "pivot_high", "pivot_low"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if zone_tol <= 0:
        raise ValueError(f"zone_tol must be > 0, got {zone_tol}")
    if min_touches < 1:
        raise ValueError(f"min_touches must be >= 1, got {min_touches}")

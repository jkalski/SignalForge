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

Complexity
----------
Sorting       : O(N log N)  via np.argsort
Gap detection : O(N)        via np.diff + boolean mask
Aggregation   : O(K)        small loop over K clusters,  K << N
Total         : O(N log N)

The only Python-level loop is the final O(K) zone-construction step, where K
is the number of clusters produced (typically tens, never thousands).

ATR tolerance
-------------
Callers should pass:  zone_tol = ATR_14 * 0.5

This keeps the module dependency-free — no indicator computation happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """A clustered support or resistance price zone."""

    kind: Literal["support", "resistance"]
    low: float          # lowest pivot price in the cluster
    high: float         # highest pivot price in the cluster
    center: float       # midpoint of [low, high]
    touches: int        # number of pivots merged into this zone
    first_ts: pd.Timestamp  # timestamp of the earliest contributing pivot
    last_ts: pd.Timestamp   # timestamp of the most recent contributing pivot

    @property
    def width(self) -> float:
        """Price width of the zone."""
        return self.high - self.low


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_zones_from_pivots(
    df: pd.DataFrame,
    zone_tol: float,
    min_touches: int = 2,
) -> List[Zone]:
    """
    Cluster pivot highs and pivot lows into support / resistance zones.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``find_pivots()``.  Required columns:
        ts, high, low, pivot_high (bool), pivot_low (bool).
        Rows must be in chronological order.
    zone_tol : float
        Maximum price distance between two pivots for them to be merged
        into the same zone.  Recommended: ATR_14 * 0.5.
    min_touches : int, default 2
        Zones with fewer pivot touches are discarded.

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

    # Resistance zones: built from pivot highs using the bar's 'high' price.
    if df["pivot_high"].any():
        mask = df["pivot_high"]
        prices = df.loc[mask, "high"].to_numpy(dtype=np.float64)
        timestamps = df.loc[mask, "ts"].to_numpy()
        zones.extend(_cluster_pivots(prices, timestamps, zone_tol, "resistance"))

    # Support zones: built from pivot lows using the bar's 'low' price.
    if df["pivot_low"].any():
        mask = df["pivot_low"]
        prices = df.loc[mask, "low"].to_numpy(dtype=np.float64)
        timestamps = df.loc[mask, "ts"].to_numpy()
        zones.extend(_cluster_pivots(prices, timestamps, zone_tol, "support"))

    # Apply minimum touch filter.
    zones = [z for z in zones if z.touches >= min_touches]

    # Primary sort: most-tested zones first; secondary: most-recent first.
    zones.sort(key=lambda z: (z.touches, z.last_ts), reverse=True)

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
    """
    Group price levels into zones by sorting then gap-detection.

    Only Python-level loop: final O(K) construction of Zone objects where
    K = number of clusters produced.
    """
    if len(prices) == 0:
        return []

    # --- Step 1: sort pivots by price ascending --------------------------
    sort_idx = np.argsort(prices, kind="stable")
    sorted_prices = prices[sort_idx]
    sorted_ts = timestamps[sort_idx]

    # --- Step 2: detect cluster boundaries via single np.diff pass -------
    # gaps[i] is True when there is a "break" between sorted_prices[i] and
    # sorted_prices[i+1], meaning they belong to different clusters.
    gaps: np.ndarray = np.diff(sorted_prices) > zone_tol  # shape (n-1,)

    # Absolute index (into sorted arrays) where each new cluster begins.
    # The first cluster always starts at 0.
    starts: np.ndarray = np.concatenate([[0], np.where(gaps)[0] + 1])
    ends: np.ndarray = np.concatenate([starts[1:], [len(sorted_prices)]])

    # --- Step 3: aggregate per-cluster stats with numpy ufuncs ----------
    # Price range: array is sorted, so min = first element, max = last.
    low_prices: np.ndarray = sorted_prices[starts]
    high_prices: np.ndarray = sorted_prices[ends - 1]
    center_prices: np.ndarray = (low_prices + high_prices) * 0.5
    touches: np.ndarray = ends - starts   # element count per cluster

    # Timestamp min/max per cluster.
    # np.minimum.reduceat is O(N) and handles datetime64 natively.
    first_ts: np.ndarray = np.minimum.reduceat(sorted_ts, starts)
    last_ts: np.ndarray = np.maximum.reduceat(sorted_ts, starts)

    # --- Step 4: build Zone objects (O(K) Python loop) -------------------
    n_clusters = len(starts)
    zones: List[Zone] = [
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

    return zones


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, zone_tol: float, min_touches: int) -> None:
    required = {"ts", "high", "low", "pivot_high", "pivot_low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if zone_tol <= 0:
        raise ValueError(f"zone_tol must be > 0, got {zone_tol}")
    if min_touches < 1:
        raise ValueError(f"min_touches must be >= 1, got {min_touches}")

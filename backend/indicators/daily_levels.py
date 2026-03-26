"""
Previous Day High / Low (PDH / PDL) zone builder.

For each completed calendar day in the DataFrame, this module emits two
Zone objects:
  - Resistance Zone at the prior day's high (kind="resistance")
  - Support    Zone at the prior day's low  (kind="support")

These act as synthetic S/R levels with a single touch count (touches=1).
Because they are just Zone objects, they slot directly into the existing
zone pipeline used by build_zones_from_pivots() and build_sd_zones().

Look-ahead safety
-----------------
Only *completed* days are included.  The current (most recent, potentially
unfinished) day is excluded so that live signals never see the current
session's high/low as a "prior day" level.

Usage
-----
    from backend.indicators.daily_levels import get_daily_levels
    pdh_pdl = get_daily_levels(df, zone_tol)
    zones = zones + pdh_pdl   # merge before zone selection
"""

from __future__ import annotations

from typing import List

import pandas as pd

from backend.indicators.zones import Zone


def get_daily_levels(
    df: pd.DataFrame,
    zone_tol: float,
) -> List[Zone]:
    """
    Return PDH and PDL zones for all completed calendar days in df.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, high, low.
        ``ts`` must be parseable as a pandas Timestamp (datetime-like or
        datetime64).
    zone_tol : float
        Passed through for API symmetry.  Used for deduplication: if two
        levels of the same kind are within zone_tol of each other, only
        the more recent one is kept.

    Returns
    -------
    List[Zone]
        One resistance Zone (PDH) + one support Zone (PDL) per completed
        calendar day, sorted by (touches asc, last_ts desc) — matching the
        S/D zone convention so fresh levels come first.
        Empty list if df is empty or contains only a single calendar day.
    """
    required = {"ts", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if df.empty:
        return []

    # Ensure ts is datetime-typed for .dt accessor.
    ts_col = pd.to_datetime(df["ts"])
    dates = ts_col.dt.date
    all_dates = sorted(dates.unique())

    # Need at least 2 distinct dates to have one completed prior day.
    if len(all_dates) < 2:
        return []

    # Current (last) date is excluded — it may be incomplete.
    completed_dates = all_dates[:-1]

    zones: List[Zone] = []
    for d in completed_dates:
        mask = dates == d
        day_df = df[mask]
        if day_df.empty:
            continue

        day_ts = ts_col[mask]
        first_ts = pd.Timestamp(day_ts.iloc[0])
        last_ts  = pd.Timestamp(day_ts.iloc[-1])

        pdh = float(day_df["high"].max())
        pdl = float(day_df["low"].min())

        if pdh <= pdl:
            continue

        zones.append(Zone(
            kind="resistance",
            low=pdh - zone_tol * 0.5,
            high=pdh + zone_tol * 0.5,
            center=pdh,
            touches=1,
            first_ts=first_ts,
            last_ts=last_ts,
        ))
        zones.append(Zone(
            kind="support",
            low=pdl - zone_tol * 0.5,
            high=pdl + zone_tol * 0.5,
            center=pdl,
            touches=1,
            first_ts=first_ts,
            last_ts=last_ts,
        ))

    # Deduplicate: if two same-kind zones are within zone_tol, keep the newer.
    zones = _deduplicate(zones, zone_tol)

    # Fresh-first (all have touches=1), then most-recent first.
    zones.sort(key=lambda z: (z.touches, -z.last_ts.value))

    return zones


def _deduplicate(zones: List[Zone], tol: float) -> List[Zone]:
    """Keep the most recent zone when two same-kind centers are within tol."""
    if not zones:
        return zones
    # Sort newest first so the first occurrence of a price cluster is kept.
    zones_sorted = sorted(zones, key=lambda z: -z.last_ts.value)
    kept: List[Zone] = []
    for z in zones_sorted:
        duplicate = any(
            k.kind == z.kind and abs(k.center - z.center) <= tol
            for k in kept
        )
        if not duplicate:
            kept.append(z)
    return kept

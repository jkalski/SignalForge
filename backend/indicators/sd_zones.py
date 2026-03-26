"""
Supply / demand zone detection.

Algorithm
---------
A supply/demand zone is identified by two components:

  Base   — 1–N candles of tight consolidation before a strong move.
            Tight means every bar's range (high - low) < ATR * base_range_mult.

  Impulse — the candle immediately after the base that leaves the area
             with conviction.  Conviction means body (|close - open|) > ATR
             * min_impulse_atr.

Zone kind
---------
  Impulse closes *above* open → demand zone (kind="support")
      Price left the base upward → institutional buying.
  Impulse closes *below* open → supply zone (kind="resistance")
      Price left the base downward → institutional selling.

Zone boundaries
---------------
  low  = lowest low  of the base candles
  high = highest high of the base candles
  center = (low + high) / 2

Touch counting
--------------
After zone formation, each subsequent bar whose low–high range overlaps
the zone [low, high] increments the touch counter.  Zones are sorted
fresh-first (touches ascending) unlike pivot S/R.

Freshness matters because supply/demand zones are consumed with each
visit; an untested zone has a higher probability of holding.

Returns
-------
List[Zone] using the same Zone dataclass as build_zones_from_pivots().
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from backend.indicators.zones import Zone


def build_sd_zones(
    df: pd.DataFrame,
    zone_tol: float,
    atr: float,
    min_impulse_atr: float = 1.5,
    max_base_bars: int = 5,
    base_range_mult: float = 0.6,
) -> List[Zone]:
    """
    Detect supply and demand zones from candle data.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, high, low, close.
        (pivot_high / pivot_low columns are ignored — not used here.)
    zone_tol : float
        Price tolerance for overlap checks (ATR * zone_atr_mult).
        Kept for API symmetry with build_zones_from_pivots; not used
        internally but available for callers.
    atr : float
        Current ATR14 of the symbol.  Used to define "tight base" and
        "strong impulse" thresholds.
    min_impulse_atr : float, default 1.5
        Impulse body must be > atr * this to qualify as a strong move.
    max_base_bars : int, default 5
        Maximum number of consecutive base candles to search for.
    base_range_mult : float, default 0.6
        A base candle's range (high - low) must be < atr * this.

    Returns
    -------
    List[Zone]
        Zones sorted by (touches asc, last_ts desc) — fresh zones first.
        Empty list if no qualifying zones found or atr <= 0.
    """
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if atr <= 0 or not np.isfinite(atr):
        return []
    if len(df) < max_base_bars + 2:
        return []

    max_base_range = atr * base_range_mult
    min_body       = atr * min_impulse_atr

    opens  = df["open"].to_numpy(dtype=np.float64)
    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts_arr = df["ts"].to_numpy()

    n = len(df)
    zones: List[Zone] = []
    # Track zone indices already emitted to avoid duplicates from
    # overlapping base windows.
    used_impulse: set = set()

    # Scan every bar as a potential impulse candle (bar i).
    # Look back up to max_base_bars to find a qualifying base.
    for i in range(1, n):
        body = abs(closes[i] - opens[i])
        if body < min_body:
            continue                    # not a strong impulse
        if i in used_impulse:
            continue

        bullish = closes[i] > opens[i]  # True → demand zone

        # Search backward for consecutive tight base bars ending at i-1.
        base_end = i - 1
        base_start = base_end

        # Walk back while bars are tight.
        while base_start > 0 and (base_end - base_start) < max_base_bars:
            j = base_start
            if (highs[j] - lows[j]) > max_base_range:
                break
            base_start -= 1
        # base_start ended one step too far back after the loop exit.
        base_start += 1

        base_len = base_end - base_start + 1
        if base_len < 1:
            continue

        zone_low  = float(lows[base_start : base_end + 1].min())
        zone_high = float(highs[base_start : base_end + 1].max())

        if zone_high <= zone_low:
            continue

        kind: str = "support" if bullish else "resistance"

        # Count how many bars after formation overlap the zone.
        future = slice(i + 1, n)
        future_lows  = lows[future]
        future_highs = highs[future]
        if len(future_lows) > 0:
            revisits = int(np.sum(
                (future_lows <= zone_high) & (future_highs >= zone_low)
            ))
        else:
            revisits = 0

        touches  = 1 + revisits   # 1 = formation bar; +N = subsequent visits
        first_ts = pd.Timestamp(ts_arr[base_start])
        last_ts  = (
            pd.Timestamp(ts_arr[i + revisits])
            if revisits > 0 and (i + revisits) < n
            else pd.Timestamp(ts_arr[i])
        )

        # --- ML features computed at formation time -----------------------
        zone_center = (zone_low + zone_high) * 0.5

        # zone_age_bars: bars from last touch to current bar.
        last_touch_idx  = (i + revisits) if revisits > 0 and (i + revisits) < n else i
        zone_age_bars   = int(n - 1 - last_touch_idx)

        # zone_reaction_avg: impulse body as ATR multiples (the formation
        # move IS the first reaction; subsequent reactions use forward closes).
        impulse_reaction = abs(closes[i] - opens[i]) / atr   # body / ATR
        reactions        = [impulse_reaction]
        failed           = 0

        if revisits > 0:
            reaction_window = 5
            for k in range(revisits):
                bar_idx = i + 1 + k
                fwd_end = min(bar_idx + reaction_window + 1, n)
                fwd_c   = closes[bar_idx + 1 : fwd_end] if bar_idx + 1 < n else np.array([])
                if len(fwd_c) == 0:
                    continue
                if bullish:
                    move = (float(np.max(fwd_c)) - zone_center) / atr
                else:
                    move = (zone_center - float(np.min(fwd_c))) / atr
                if move >= 0.5:
                    reactions.append(move)
                else:
                    failed += 1

        zone_reaction_avg = float(np.mean(reactions))

        # volume_at_zone: mean volume of base candles (institutional footprint).
        if "volume" in df.columns:
            base_vols  = pd.to_numeric(
                df["volume"].iloc[base_start : base_end + 1], errors="coerce"
            ).to_numpy(dtype=np.float64)
            valid_vols = base_vols[np.isfinite(base_vols) & (base_vols > 0)]
            volume_at_zone = float(np.mean(valid_vols)) if len(valid_vols) > 0 else 0.0
        else:
            volume_at_zone = 0.0

        # strength_score — import inline to avoid circular dependency.
        from backend.indicators.zones import _compute_strength  # noqa: PLC0415
        import math as _math

        touch_score    = min(touches / 6.0, 1.0)
        reaction_score = min(zone_reaction_avg / 3.0, 1.0)
        age_decay      = _math.exp(-zone_age_bars / 100.0)
        failure_mult   = max(0.0, 1.0 - failed * 0.15)
        strength_score = round(min(
            0.35 * touch_score + 0.35 * reaction_score
            + 0.20 * age_decay + 0.10 * failure_mult,
            1.0,
        ), 4)

        zones.append(Zone(
            kind=kind,
            low=zone_low,
            high=zone_high,
            center=zone_center,
            touches=touches,
            first_ts=first_ts,
            last_ts=last_ts,
            strength_score=strength_score,
            zone_age_bars=zone_age_bars,
            zone_reaction_avg=zone_reaction_avg,
            failed_tests=failed,
            volume_at_zone=volume_at_zone,
        ))
        used_impulse.add(i)

    # Remove near-duplicate zones (same kind, centers within zone_tol).
    zones = _deduplicate(zones, zone_tol)

    # Fresh zones first (low touches), then most-recent.
    zones.sort(key=lambda z: (z.touches, -z.last_ts.value))

    return zones


def _deduplicate(zones: List[Zone], tol: float) -> List[Zone]:
    """Remove zones of the same kind whose centers are within tol of each other."""
    if not zones:
        return zones
    kept: List[Zone] = []
    for z in zones:
        duplicate = False
        for k in kept:
            if k.kind == z.kind and abs(k.center - z.center) <= tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(z)
    return kept

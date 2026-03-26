"""
Inside Bar and Outside Bar (Engulfing) pattern detection.

Inside Bar
----------
An Inside Bar pattern consists of a "Mother Bar" whose high and low completely
engulf the subsequent candle(s).  The inside bar(s) reflect consolidation and
reduced volatility after a large move.

Signal fires when the current bar breaks *out of* a recently-formed inside bar
pattern:
  inside_bar_long         — close > mother_bar_high (single inside bar breakout)
  inside_bar_short        — close < mother_bar_low  (single inside bar breakout)
  double_inside_bar_long  — same, but 2+ consecutive inside bars formed (stronger)
  double_inside_bar_short — same, bearish direction

Multiple consecutive inside bars within the same mother bar range signal stronger
consolidation and tend to produce more explosive breakouts when they resolve.

Outside Bar (Engulfing)
-----------------------
An outside bar is a single candlestick whose high AND low completely engulf the
prior bar — equivalent to a bullish or bearish engulfing pattern:
  outside_bar_long  — outside bar closes bullish (buyers overwhelmed sellers)
  outside_bar_short — outside bar closes bearish (sellers overwhelmed buyers)

Both patterns perform best on higher timeframes (Daily, Weekly).  Lower-timeframe
false-positives are common; callers may apply their own timeframe gate.

Note: these patterns are primarily designed for individual equities.  On broad
index instruments, noise is higher and false-breakout rates increase — apply
additional confluence filters (trend, volume, zone proximity) before acting.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


_LOOKBACK = 10   # bars to scan back for a mother bar formation


def detect_inside_bar(
    df: pd.DataFrame,
    lookback: int = _LOOKBACK,
) -> Optional[Dict]:
    """
    Detect a breakout from a recent inside bar pattern on the last bar.

    Scans backwards through the last ``lookback`` bars for the most recent
    valid mother bar + consecutive inside bar(s) formation, then checks
    whether the current (last) bar closes outside the mother bar's range.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, high, low, close.
    lookback : int, default 10
        How many bars back to search for a mother bar formation.

    Returns
    -------
    dict or None
        If a breakout is detected:
            type            : "inside_bar_long" | "inside_bar_short"
                              or "double_inside_bar_long" | "double_inside_bar_short"
            zone_center     : float  — midpoint of the mother bar's range
            touches         : 1
            timestamp       : df["ts"].iloc[-1]
            inside_bar_count: int   — number of consecutive inside bars (1 = single,
                                      2+ = double/triple, shown in type name)
        None if no qualifying breakout found.
    """
    required = {"ts", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 3:
        return None

    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts_arr = df["ts"].to_numpy()

    n          = len(df)
    curr_close = closes[n - 1]
    curr_ts    = ts_arr[n - 1]

    # scan_end is the last confirmed bar before the current breakout bar.
    scan_end   = n - 2

    # Walk backwards: each candidate mother_idx is tested to see whether
    # all bars from mother_idx+1 through scan_end are inside bars.
    for mother_idx in range(scan_end - 1, max(n - lookback - 1, 0), -1):
        mother_high  = highs[mother_idx]
        mother_low   = lows[mother_idx]
        mother_range = mother_high - mother_low

        if mother_range <= 0:
            continue

        # Count consecutive inside bars immediately after the mother bar.
        ib_count = 0
        for ib_idx in range(mother_idx + 1, scan_end + 1):
            if highs[ib_idx] < mother_high and lows[ib_idx] > mother_low:
                ib_count += 1
            else:
                break   # containment broken — this candidate is invalid

        if ib_count == 0:
            continue

        # Valid pattern found.  Check if the current bar breaks out.
        zone_center = (mother_high + mother_low) * 0.5
        base_type   = "double_inside_bar" if ib_count >= 2 else "inside_bar"

        if curr_close > mother_high:
            return {
                "type":             f"{base_type}_long",
                "zone_center":      zone_center,
                "touches":          1,
                "timestamp":        curr_ts,
                "inside_bar_count": ib_count,
            }

        if curr_close < mother_low:
            return {
                "type":             f"{base_type}_short",
                "zone_center":      zone_center,
                "touches":          1,
                "timestamp":        curr_ts,
                "inside_bar_count": ib_count,
            }

    return None


def detect_outside_bar(
    df: pd.DataFrame,
) -> Optional[Dict]:
    """
    Detect a bullish or bearish outside bar (engulfing) on the last bar.

    The outside bar's high must exceed the previous bar's high AND its low
    must be below the previous bar's low.  Direction is determined by whether
    the bar closes bullish (green) or bearish (red).  Doji outside bars
    (open == close) are ignored — no directional conviction.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, high, low, close.

    Returns
    -------
    dict or None
        If detected:
            type       : "outside_bar_long" | "outside_bar_short"
            zone_center: float  — midpoint of the outside bar's range
            touches    : 1
            timestamp  : df["ts"].iloc[-1]
        None if no qualifying outside bar found.
    """
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 2:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    curr_high  = float(curr["high"])
    curr_low   = float(curr["low"])
    curr_open  = float(curr["open"])
    curr_close = float(curr["close"])
    prev_high  = float(prev["high"])
    prev_low   = float(prev["low"])

    # Outside bar: current range completely engulfs the prior bar.
    if curr_high <= prev_high or curr_low >= prev_low:
        return None

    zone_center = (curr_high + curr_low) * 0.5
    curr_ts     = curr["ts"]

    if curr_close > curr_open:
        return {
            "type":        "outside_bar_long",
            "zone_center": zone_center,
            "touches":     1,
            "timestamp":   curr_ts,
        }

    if curr_close < curr_open:
        return {
            "type":        "outside_bar_short",
            "zone_center": zone_center,
            "touches":     1,
            "timestamp":   curr_ts,
        }

    return None   # doji — no directional conviction


def detect_ath_breakout(
    df: pd.DataFrame,
    zone_tol: float,
) -> Optional[Dict]:
    """
    Detect an All-Time High breakout on the last bar.

    An ATH breakout fires when the current bar's close exceeds the highest
    high across all prior bars in the dataset.  This condition eliminates
    overhead supply entirely — no trapped sellers remain above price —
    creating conditions for rapid, target-exceeding gains.

    A minimum clearance of ``zone_tol`` above the prior ATH is required to
    filter marginal closes that merely graze prior resistance without
    conviction.

    Note: the "ATH" here is relative to the candle history provided.  For a
    true all-time high check, callers should supply the maximum possible
    historical window (e.g. daily candles going back years).  On shorter
    windows this fires when price clears the lookback-period high.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, high, close.
    zone_tol : float
        Minimum price clearance above the prior ATH (typically ATR * 0.5).

    Returns
    -------
    dict or None
        If detected:
            type       : "ath_breakout_long"
            zone_center: float  — the ATH level that was cleared
            touches    : 1
            timestamp  : df["ts"].iloc[-1]
        None if the last bar is not breaking to a new ATH.
    """
    required = {"ts", "high", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 2:
        return None

    highs  = df["high"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts_arr = df["ts"].to_numpy()

    n          = len(df)
    curr_close = closes[n - 1]
    curr_ts    = ts_arr[n - 1]

    # ATH = max high of all bars prior to the current bar.
    prior_ath = float(np.max(highs[: n - 1]))

    if curr_close > prior_ath + zone_tol:
        return {
            "type":        "ath_breakout_long",
            "zone_center": prior_ath,
            "touches":     1,
            "timestamp":   curr_ts,
        }

    return None

"""
Fair Value Gap (FVG) detection.

An FVG is a 3-candle imbalance pattern where price leaves a visible gap
between candle[i-2] and candle[i], bypassing the wick of candle[i-1].

Pattern definitions
-------------------
  Bullish FVG : candle[i-2].high < candle[i].low
                A gap exists *above* candle[i-2].  Price left upward fast.
                When price later re-enters from above (current bar's low
                enters the gap), it signals a potential long entry.

  Bearish FVG : candle[i-2].low > candle[i].high
                A gap exists *below* candle[i-2].  Price left downward fast.
                When price later re-enters from below (current bar's high
                enters the gap), it signals a potential short entry.

Signal conditions (evaluated on the most recent closed bar)
------------------------------------------------------------
  fvg_long  — last bar's low  dips into a recent bullish FVG and close > gap midpoint
  fvg_short — last bar's high pops into a recent bearish FVG and close < gap midpoint

ATR filter
----------
Only FVGs whose gap width > atr * min_gap_atr are considered (avoids noise).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_MIN_GAP_ATR = 0.3   # gap must be at least 30 % of ATR to qualify
_LOOKBACK    = 15    # number of bars to scan for FVG formation


def detect_fvg(
    df: pd.DataFrame,
    zone_tol: float,
    atr: float,
    min_gap_atr: float = _MIN_GAP_ATR,
    lookback: int = _LOOKBACK,
) -> Optional[Dict]:
    """
    Detect whether the most recent bar is entering a recent Fair Value Gap.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, high, low, close.
    zone_tol : float
        Price tolerance (ATR * zone_atr_mult); kept for API symmetry.
    atr : float
        Current ATR14.  Used to filter noise-level FVGs.
    min_gap_atr : float, default 0.3
        Minimum FVG width as a fraction of ATR.
    lookback : int, default 15
        How many bars back to search for FVG formations.

    Returns
    -------
    dict or None
        If an FVG entry is detected:
            type       : "fvg_long" | "fvg_short"
            zone_center: float  — midpoint of the FVG gap
            touches    : 1
            timestamp  : value of df["ts"].iloc[-1]
        None if no qualifying FVG entry found.
    """
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 3 or atr <= 0:
        return None

    min_gap = atr * min_gap_atr

    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts_arr = df["ts"].to_numpy()

    n        = len(df)
    last_bar = n - 1
    curr_low  = lows[last_bar]
    curr_high = highs[last_bar]
    curr_close = closes[last_bar]
    curr_ts    = ts_arr[last_bar]

    # Scan recent bars (excluding the last bar which is the signal bar).
    scan_start = max(2, n - lookback)  # need i-2 and i to be valid

    # Collect all qualifying FVGs from the lookback window.
    bullish_fvgs: List[Tuple[float, float]] = []  # (gap_low, gap_high)
    bearish_fvgs: List[Tuple[float, float]] = []

    for i in range(scan_start, last_bar):   # i is the right candle of the 3-bar pattern
        gap_low  = highs[i - 2]
        gap_high = lows[i]
        if gap_high > gap_low and (gap_high - gap_low) >= min_gap:
            # Valid bullish FVG: candle[i-2].high < candle[i].low
            bullish_fvgs.append((gap_low, gap_high))
            continue

        gap_low  = highs[i]
        gap_high = lows[i - 2]
        if gap_high > gap_low and (gap_high - gap_low) >= min_gap:
            # Valid bearish FVG: candle[i-2].low > candle[i].high
            bearish_fvgs.append((gap_low, gap_high))

    # Check if the current bar is entering a bullish FVG from above (long).
    for (gl, gh) in bullish_fvgs:
        midpoint = (gl + gh) * 0.5
        # Price dips into the gap zone and closes above midpoint.
        if curr_low <= gh and curr_low >= gl and curr_close > midpoint:
            return {
                "type":        "fvg_long",
                "zone_center": midpoint,
                "touches":     1,
                "timestamp":   curr_ts,
            }

    # Check if the current bar is entering a bearish FVG from below (short).
    for (gl, gh) in bearish_fvgs:
        midpoint = (gl + gh) * 0.5
        # Price pops into the gap zone and closes below midpoint.
        if curr_high >= gl and curr_high <= gh and curr_close < midpoint:
            return {
                "type":        "fvg_short",
                "zone_center": midpoint,
                "touches":     1,
                "timestamp":   curr_ts,
            }

    return None

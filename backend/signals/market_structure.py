"""
Break of Structure (BOS) and Change of Character (ChoCh) detector.

Market structure is defined by the sequence of confirmed swing highs and
swing lows.  BOS and ChoCh are identified by watching how price interacts
with these levels.

Definitions
-----------
  BOS up   — close clears the most recent confirmed swing high.
             Signals trend continuation (higher highs forming).

  BOS down — close breaks the most recent confirmed swing low.
             Signals trend continuation (lower lows forming).

  ChoCh up — BOS up occurs after the market had printed ≥2 lower swing
             highs in a row (i.e., was previously in a downtrend).
             Signals a potential reversal / change of character to bullish.

  ChoCh down — BOS down occurs after the market had printed ≥2 higher
               swing lows (previously uptrending).
               Signals a potential reversal to bearish.

Inputs
------
Requires ``pivot_high`` and ``pivot_low`` boolean columns in df.
These are computed in pipeline Step 4 / backtest _run_bar() via find_pivots().

Confirmation
------------
The last bar's close must exceed the swing level by > zone_tol to avoid
false BOS on marginal closes (same buffer used in breakout detection).

Priority
--------
ChoCh (potential reversal) is preferred over BOS (continuation) when both
would fire, because ChoCh represents a more significant structural change.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def detect_bos(
    df: pd.DataFrame,
    zone_tol: float,
) -> Optional[Dict]:
    """
    Detect a Break of Structure or Change of Character on the last bar.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, high, low, close, pivot_high, pivot_low.
    zone_tol : float
        Minimum price clearance above/below the swing level for confirmation.

    Returns
    -------
    dict or None
        If a BOS / ChoCh is detected:
            type       : "bos_up" | "bos_down" | "choch_up" | "choch_down"
            zone_center: float  — the broken swing level
            touches    : int    — number of pivot touches used for level
            timestamp  : value of df["ts"].iloc[-1]
        None if no qualifying event found.
    """
    required = {"ts", "high", "low", "close", "pivot_high", "pivot_low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 5:
        return None

    curr      = df.iloc[-1]
    curr_close = float(curr["close"])
    curr_ts    = curr["ts"]

    # Collect confirmed swing highs and lows (exclude the last bar).
    hist = df.iloc[:-1]

    swing_highs: List[float] = hist.loc[hist["pivot_high"] == True, "high"].tolist()
    swing_lows:  List[float] = hist.loc[hist["pivot_low"]  == True, "low"].tolist()

    # Need at least 2 swing levels of each type to assess trend sequence.
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    last_sh  = swing_highs[-1]   # most recent swing high
    prev_sh  = swing_highs[-2]   # second most recent swing high
    last_sl  = swing_lows[-1]    # most recent swing low
    prev_sl  = swing_lows[-2]    # second most recent swing low

    event_type:  Optional[str]   = None
    broken_level: Optional[float] = None

    # --- Break of Structure UP ---
    if curr_close > last_sh + zone_tol:
        # Were the last 2 swing highs descending (downtrend → reversal)?
        if last_sh < prev_sh:
            # BOS up after ≥2 lower highs → Change of Character
            event_type   = "choch_up"
        else:
            # BOS up in an ongoing uptrend → continuation
            event_type   = "bos_up"
        broken_level = last_sh

    # --- Break of Structure DOWN ---
    elif curr_close < last_sl - zone_tol:
        # Were the last 2 swing lows ascending (uptrend → reversal)?
        if last_sl > prev_sl:
            # BOS down after ≥2 higher lows → Change of Character
            event_type   = "choch_down"
        else:
            # BOS down in an ongoing downtrend → continuation
            event_type   = "bos_down"
        broken_level = last_sl

    if event_type is None:
        return None

    return {
        "type":        event_type,
        "zone_center": broken_level,
        "touches":     1,
        "timestamp":   curr_ts,
    }

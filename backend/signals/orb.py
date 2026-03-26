"""
Opening Range Breakout (ORB) signal detector.

The Opening Range is defined by the first full candle of the regular trading
session (09:30 ET = 14:30 UTC for US equities).

  ORH = first session bar's high
  ORL = first session bar's low

A breakout is confirmed when a subsequent bar's close clears the range
by at least zone_tol (preventing false breakouts on marginal closes).

Session awareness
-----------------
The detector groups bars by calendar date.  Within each session, the first
bar at or after 14:30 UTC defines the opening range.  The signal can only
fire once per session (the earliest qualifying bar wins).  If the opening
range bar is the last bar in df, no signal is emitted (there is no
subsequent bar to trigger on).

Signal types
------------
  orb_long  — close > ORH + zone_tol (bullish breakout above range)
  orb_short — close < ORL - zone_tol (bearish breakdown below range)

These are most relevant for liquid index ETFs (SPY, QQQ) and high-volume
large-caps where the open prints a tight, well-defined range.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import numpy as np


_SESSION_OPEN_HOUR_UTC = 14   # 09:30 ET = 14:30 UTC  (whole-hour guard)
_SESSION_OPEN_MIN_UTC  = 30


def detect_orb(
    df: pd.DataFrame,
    atr: float,
    zone_tol: float,
) -> Optional[Dict]:
    """
    Detect an Opening Range Breakout on the most recent closed bar.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, high, low, close.
        ``ts`` must be UTC-aware or naive datetime-like (treated as UTC).
    atr : float
        Current ATR14.  Retained for API consistency; zone_tol is used for
        breakout confirmation.
    zone_tol : float
        Price buffer above ORH / below ORL required for confirmation.

    Returns
    -------
    dict or None
        If an ORB is detected on the last bar:
            type       : "orb_long" | "orb_short"
            zone_center: float  — midpoint of the opening range
            touches    : 1
            timestamp  : value of df["ts"].iloc[-1]
        None if no qualifying ORB found.
    """
    required = {"ts", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 2:
        return None

    ts_series = pd.to_datetime(df["ts"])

    # Identify the current session date (date of the last bar).
    last_ts = ts_series.iloc[-1]
    session_date = last_ts.date()

    # Find all bars belonging to the current session date.
    session_mask = ts_series.dt.date == session_date
    session_df = df[session_mask]
    session_ts = ts_series[session_mask]

    if len(session_df) < 2:
        return None

    # Find the opening range bar: first bar at or after 14:30 UTC.
    orb_mask = (
        (session_ts.dt.hour > _SESSION_OPEN_HOUR_UTC) |
        (
            (session_ts.dt.hour == _SESSION_OPEN_HOUR_UTC) &
            (session_ts.dt.minute >= _SESSION_OPEN_MIN_UTC)
        )
    )
    orb_candidates = session_df[orb_mask]
    if orb_candidates.empty:
        return None

    orb_bar = orb_candidates.iloc[0]
    orh = float(orb_bar["high"])
    orl = float(orb_bar["low"])
    or_center = (orh + orl) * 0.5

    # The signal bar is the last bar — it must NOT be the ORB bar itself.
    if df.index[-1] == orb_candidates.index[0]:
        return None

    curr = df.iloc[-1]
    curr_close = float(curr["close"])
    curr_ts    = curr["ts"]

    if curr_close > orh + zone_tol:
        return {
            "type":        "orb_long",
            "zone_center": or_center,
            "touches":     1,
            "timestamp":   curr_ts,
        }

    if curr_close < orl - zone_tol:
        return {
            "type":        "orb_short",
            "zone_center": or_center,
            "touches":     1,
            "timestamp":   curr_ts,
        }

    return None

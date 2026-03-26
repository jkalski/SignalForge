"""
VWAP Reclaim signal detector.

A VWAP reclaim occurs when price crosses the session VWAP with a
conviction candle body, signalling a potential change in intraday bias.

Signal conditions (evaluated on the last two closed bars)
---------------------------------------------------------
  vwap_reclaim_long  — prev close < vwap AND curr close > vwap
                       AND curr body (|close - open|) > atr * body_atr_min

  vwap_reclaim_short — prev close > vwap AND curr close < vwap
                       AND curr body > atr * body_atr_min

The body filter prevents whipsaw crossings on indecision candles.

Requires
--------
  vwap_session column must already be present in df (populated in pipeline
  Step 9 / backtest _run_bar() before this is called).
  atr_14 column (or a scalar atr passed by the caller) for body filtering.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


_BODY_ATR_MIN = 0.25   # minimum body size relative to ATR to confirm reclaim


def detect_vwap_reclaim(
    df: pd.DataFrame,
    atr: float,
    body_atr_min: float = _BODY_ATR_MIN,
) -> Optional[Dict]:
    """
    Detect a VWAP reclaim on the most recent closed bar.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, close, vwap_session.
        Must have at least 2 rows.
    atr : float
        Current ATR14 value for the body conviction filter.
    body_atr_min : float, default 0.25
        Minimum body size as fraction of ATR required for signal.

    Returns
    -------
    dict or None
        If a reclaim is detected:
            type       : "vwap_reclaim_long" | "vwap_reclaim_short"
            zone_center: float  — the VWAP level at the time of crossing
            touches    : 1
            timestamp  : value of df["ts"].iloc[-1]
        None if no qualifying reclaim found.
    """
    required = {"ts", "open", "close", "vwap_session"}
    if not required.issubset(df.columns):
        return None   # vwap_session not always present (volume-gated)
    if len(df) < 2 or atr <= 0:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    curr_close = float(curr["close"])
    curr_open  = float(curr["open"])
    prev_close = float(prev["close"])

    vwap = float(curr["vwap_session"])
    if not np.isfinite(vwap) or vwap <= 0:
        return None

    body = abs(curr_close - curr_open)
    min_body = atr * body_atr_min

    if body < min_body:
        return None

    ts = curr["ts"]

    # Long: price reclaims VWAP from below.
    if prev_close < vwap and curr_close > vwap:
        return {
            "type":        "vwap_reclaim_long",
            "zone_center": vwap,
            "touches":     1,
            "timestamp":   ts,
        }

    # Short: price loses VWAP from above.
    if prev_close > vwap and curr_close < vwap:
        return {
            "type":        "vwap_reclaim_short",
            "zone_center": vwap,
            "touches":     1,
            "timestamp":   ts,
        }

    return None

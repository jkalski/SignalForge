"""
RSI Divergence detector — confluence filter.

Divergence is a disagreement between price action and momentum (RSI).
It does NOT generate a standalone signal event; instead it returns a
string label that callers use to award a confluence bonus in score_setup().

Divergence types
----------------
  bullish — price makes a lower low, but RSI makes a higher low.
            Momentum is strengthening even as price dips → potential long.

  bearish — price makes a higher high, but RSI makes a lower high.
            Momentum is weakening even as price rises → potential short.

Detection method
----------------
1. Find the last 2 confirmed pivot lows  (via ``pivot_low``  column) in the
   lookback window and compare both their price lows and their RSI values.
2. Find the last 2 confirmed pivot highs (via ``pivot_high`` column) in the
   lookback window and compare both their price highs and their RSI values.

Requires
--------
  rsi_14     column — already computed in pipeline Step 1 / backtest _run_bar().
  pivot_high column — computed in Step 4.
  pivot_low  column — computed in Step 4.

Usage
-----
    from backend.signals.divergence import detect_rsi_divergence
    rsi_div = detect_rsi_divergence(df)          # "bullish" | "bearish" | None
    aligned = (
        (rsi_div == "bullish" and direction == "long") or
        (rsi_div == "bearish" and direction == "short")
    )
    # Pass aligned=True to score_setup() for +8 pts divergence bucket.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


_LOOKBACK = 30   # bars to search for divergence pivot pairs


def detect_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = _LOOKBACK,
) -> Optional[str]:
    """
    Detect RSI divergence in the lookback window ending at the last bar.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: high, low, rsi_14, pivot_high, pivot_low.
    lookback : int, default 30
        Number of bars (including current) to search for pivot pairs.

    Returns
    -------
    "bullish", "bearish", or None.
    """
    required = {"high", "low", "rsi_14", "pivot_high", "pivot_low"}
    if not required.issubset(df.columns):
        return None   # rsi_14 not available in all contexts (e.g. backtest window)
    if len(df) < 5:
        return None

    window = df.iloc[-lookback:]

    # --- Bearish divergence: price higher high, RSI lower high ---
    ph_window = window[window["pivot_high"] == True]
    if len(ph_window) >= 2:
        last_ph  = ph_window.iloc[-1]
        prev_ph  = ph_window.iloc[-2]
        price_hh = float(last_ph["high"]) > float(prev_ph["high"])
        rsi_lh   = float(last_ph["rsi_14"]) < float(prev_ph["rsi_14"])
        if price_hh and rsi_lh:
            return "bearish"

    # --- Bullish divergence: price lower low, RSI higher low ---
    pl_window = window[window["pivot_low"] == True]
    if len(pl_window) >= 2:
        last_pl  = pl_window.iloc[-1]
        prev_pl  = pl_window.iloc[-2]
        price_ll = float(last_pl["low"]) < float(prev_pl["low"])
        rsi_hl   = float(last_pl["rsi_14"]) > float(prev_pl["rsi_14"])
        if price_ll and rsi_hl:
            return "bullish"

    return None

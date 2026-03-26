"""
Overnight / session gap signal detector.

Detects price gaps between the previous session's closing bar and the
current session's opening bar.  These are structurally different from
Fair Value Gaps (intrabar imbalances) — a session gap is a single price
discontinuity created by after-hours news, earnings, or macro events.

Gap classification
------------------
  gap_fade_short — gap UP on the open, first session bar closes BELOW its open.
                   Sellers stepped in immediately → high-probability fade short.

  gap_fade_long  — gap DOWN on the open, first session bar closes ABOVE its open.
                   Buyers absorbed the gap → high-probability fade long.

  gap_go_short   — gap DOWN on the open, first session bar holds BELOW prior close.
                   Gap is confirmed → continuation short.

  gap_go_long    — gap UP on the open, first session bar holds ABOVE prior close.
                   Gap is confirmed → continuation long.

Evaluation window
-----------------
The signal is only evaluated on the session-open bar itself OR within
``max_bars_from_open`` bars of it.  Beyond that, the gap information is
stale and should instead show up as a zone level (PDH/PDL or FVG).

Gap size filter
---------------
Only gaps whose absolute size > atr * min_gap_atr are considered.
This eliminates micro-gaps caused by bid/ask spread differences.

Session detection
-----------------
A new session starts at the first bar whose timestamp is >= 14:30 UTC
(09:30 ET) that follows a bar whose timestamp was < 14:00 UTC (i.e. the
prior session's close).  The previous session's last close is the close of
the last bar before the session gap.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


_MIN_GAP_ATR       = 0.5   # gap must be >= 50 % of ATR
_MAX_BARS_FROM_OPEN = 1    # only fire on session-open bar or 1 bar after


def detect_gap(
    df: pd.DataFrame,
    atr: float,
    min_gap_atr: float = _MIN_GAP_ATR,
    max_bars_from_open: int = _MAX_BARS_FROM_OPEN,
) -> Optional[Dict]:
    """
    Detect a gap-fade or gap-continuation signal on the most recent bar.

    Parameters
    ----------
    df : pd.DataFrame
        Candle DataFrame in chronological order.
        Required columns: ts, open, high, low, close.
        ``ts`` must be datetime-like (UTC or naive treated as UTC).
    atr : float
        Current ATR14.  Used to define the minimum qualifying gap size.
    min_gap_atr : float, default 0.5
        Minimum gap size as a fraction of ATR.
    max_bars_from_open : int, default 1
        Maximum number of bars after the session-open bar on which the
        signal is still valid.

    Returns
    -------
    dict or None
        If a gap signal is detected:
            type       : "gap_fade_short" | "gap_fade_long" |
                         "gap_go_short"   | "gap_go_long"
            zone_center: float  — midpoint of the gap (prev_close + curr_open) / 2
            gap_size   : float  — absolute gap size in price
            touches    : 1
            timestamp  : value of df["ts"].iloc[-1]
        None if no qualifying gap found on the last bar.
    """
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < 3 or atr <= 0:
        return None

    min_gap = atr * min_gap_atr

    ts_series = pd.to_datetime(df["ts"])
    opens  = df["open"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    n      = len(df)

    last_idx = n - 1
    curr_ts  = df["ts"].iloc[-1]

    # ── Find the most recent session-open bar ────────────────────────────────
    # A session-open bar is the FIRST bar of each calendar date.
    # Using date-based detection (not UTC hour) handles DST automatically:
    #   EDT (summer): market opens 13:30 UTC
    #   EST (winter): market opens 14:30 UTC
    dates = ts_series.dt.date.to_numpy()

    session_open_idx: Optional[int] = None

    # Search backward from last bar within max_bars_from_open range.
    for offset in range(max_bars_from_open + 1):
        idx = last_idx - offset
        if idx < 1:
            break
        # This bar is the session open if no earlier bar shares the same date.
        if idx == 0 or dates[idx] != dates[idx - 1]:
            # Verify there is also a prior date (i.e., a previous session close).
            # Walk back to find the last bar of the previous date.
            prev_idx = idx - 1
            while prev_idx >= 0 and dates[prev_idx] == dates[idx]:
                prev_idx -= 1
            if prev_idx >= 0:
                session_open_idx = idx
                break

    if session_open_idx is None or session_open_idx < 1:
        return None

    # Previous session's last bar = last bar of the prior date.
    prev_session_end = session_open_idx - 1
    # (dates[prev_session_end] != dates[session_open_idx] by construction above)

    # ── Measure the gap ──────────────────────────────────────────────────────
    prev_close  = closes[prev_session_end]
    session_open_price = opens[session_open_idx]
    session_open_close = closes[session_open_idx]

    gap_size = session_open_price - prev_close   # positive = gap up, negative = gap down

    if abs(gap_size) < min_gap:
        return None

    gap_center = (prev_close + session_open_price) * 0.5
    gap_up   = gap_size > 0
    gap_down = gap_size < 0

    # ── Classify: fade vs continuation ──────────────────────────────────────
    event_type: Optional[str] = None

    if gap_up:
        if session_open_close < session_open_price:
            # Opened up, closed below open → immediate rejection → fade short
            event_type = "gap_fade_short"
        elif session_open_close > prev_close:
            # Opened up, held above prior close → continuation long
            event_type = "gap_go_long"

    elif gap_down:
        if session_open_close > session_open_price:
            # Opened down, closed above open → immediate recovery → fade long
            event_type = "gap_fade_long"
        elif session_open_close < prev_close:
            # Opened down, held below prior close → continuation short
            event_type = "gap_go_short"

    if event_type is None:
        return None

    return {
        "type":        event_type,
        "zone_center": gap_center,
        "gap_size":    round(abs(gap_size), 4),
        "touches":     1,
        "timestamp":   curr_ts,
    }

"""
VWAP indicators: session-reset VWAP and anchored VWAP.

Typical price
-------------
All functions use  tp = (high + low + close) / 3  as the price input.

Volume handling
---------------
Both ``add_session_vwap`` and ``add_anchored_vwap`` are robust to missing
or zero volume:

  - ``volume`` column absent entirely    → output column is all-NaN.
  - ``volume`` present but zero / NaN    → that bar contributes 0 weight;
    VWAP is computed from positive-volume bars only.
  - ``volume`` all-zero / NaN for a
    session or anchor slice              → output is NaN for that range
                                           (no valid volume to weight).

Session VWAP
------------
Resets at each calendar-day boundary derived from the ``ts`` column.
Suitable for intraday timeframes (1m, 5m, 15m, 1h).

For daily-or-higher timeframes each session contains a single bar and the
result equals the bar's typical price.  Callers should gate on timeframe
before using ``vwap_session`` for confluence.

Anchored VWAP
-------------
Cumulative VWAP from a caller-supplied anchor bar index to the end of the
DataFrame.  Bars before the anchor receive NaN.  Anchor selection is
typically the output of ``get_last_major_pivot_index``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``vwap_session`` — a VWAP that resets at each calendar-day boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Required columns: ``ts`` (datetime-like), ``high``, ``low``, ``close``.
        Optional: ``volume``.  If absent, ``vwap_session`` is all-NaN.
    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``vwap_session`` (float) added.

    Notes
    -----
    Session boundaries are the UTC calendar dates of ``ts``.  No
    market-hours or holiday correction is applied.  If ``ts`` stores
    UTC timestamps but the exchange trades in Eastern Time, callers
    should convert to ET before calling this function to avoid sessions
    splitting at the wrong boundary.
    """
    _validate_price_cols(df)
    out = df.copy()
    tp = (out["high"] + out["low"] + out["close"]) / 3.0

    if "volume" not in out.columns:
        out["vwap_session"] = np.nan
        return out

    vol = out["volume"].astype(float)
    # Treat zero / negative volume as missing (NaN); then fill with 0 so that
    # those bars add 0 to both numerator and denominator (= skip without
    # breaking the cumulative sum).
    vol_safe = vol.where(vol > 0).fillna(0.0)
    tp_vol = (tp * vol.where(vol > 0)).fillna(0.0)  # 0 contribution for bad-vol bars

    session_key = out["ts"].dt.date  # calendar date → session group key

    # cumsum within each day group — fully vectorized via pandas groupby.
    cum_tp_vol = tp_vol.groupby(session_key).transform("cumsum")
    cum_vol = vol_safe.groupby(session_key).transform("cumsum")

    # NaN where no positive volume has accumulated yet in the session.
    out["vwap_session"] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return out


def add_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.DataFrame:
    """
    Append ``vwap_anchored`` — a VWAP cumulated from *anchor_idx* to the last bar.

    Rows before *anchor_idx* receive NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Required columns: ``high``, ``low``, ``close``.
        Optional: ``volume``.  If absent, ``vwap_anchored`` is all-NaN.
    anchor_idx : int
        Row position (iloc-based).  Negative integers are resolved from the
        tail (``-1`` anchors at the last bar, ``-n`` at the n-th from end).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``vwap_anchored`` (float) added.

    Raises
    ------
    ValueError
        If *anchor_idx* is out of range after resolving negatives, or
        required columns are missing.
    """
    _validate_price_cols(df)

    n = len(df)
    idx = anchor_idx if anchor_idx >= 0 else n + anchor_idx
    if idx < 0 or idx >= n:
        raise ValueError(
            f"anchor_idx={anchor_idx!r} resolves to {idx}, "
            f"out of range [0, {n - 1}]."
        )

    out = df.copy()
    tp = (out["high"] + out["low"] + out["close"]) / 3.0

    if "volume" not in out.columns:
        out["vwap_anchored"] = np.nan
        return out

    vol = out["volume"].astype(float)
    vol_safe = vol.where(vol > 0).fillna(0.0)  # 0 for invalid bars

    # Slice from anchor onward; cumsum naturally starts fresh at anchor_idx.
    tp_slice = tp.iloc[idx:]
    vol_slice = vol_safe.iloc[idx:]
    tp_vol_slice = (tp_slice * vol_slice)

    cum_tp_vol = tp_vol_slice.cumsum()
    cum_vol = vol_slice.cumsum()

    avwap_values = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)

    # Build full-length column: NaN before anchor, AVWAP from anchor onward.
    avwap_full = pd.Series(np.nan, index=out.index, dtype=float)
    avwap_full.iloc[idx:] = avwap_values

    out["vwap_anchored"] = avwap_full
    return out


def get_last_major_pivot_index(
    df: pd.DataFrame,
    kind: Literal["low", "high"] = "low",
    lookback_pivots: int = 5,
    prominence_horizon: int = 20,
) -> Optional[int]:
    """
    Return the iloc index of the most prominent recent pivot for AVWAP anchoring.

    Prominence is proxied by the **swing distance**: the distance from the
    pivot's price to the extremum of the following ``prominence_horizon`` bars.
    A larger swing means the pivot triggered a more significant reversal and
    is therefore a better AVWAP anchor.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pivot_low`` and/or ``pivot_high`` boolean columns
        (output of ``find_pivots()``), plus ``low``, ``high``, ``close``.
    kind : {"low", "high"}, default "low"
        ``"low"``  — most significant swing low (bullish / support context).
        ``"high"`` — most significant swing high (bearish / resistance context).
    lookback_pivots : int, default 5
        Number of most-recent pivots to consider as candidates.
        Increase for sparser data (daily+).
    prominence_horizon : int, default 20
        Bars forward from the pivot used to measure the resulting swing.

    Returns
    -------
    int or None
        iloc position of the selected pivot, or ``None`` if no pivots exist.

    Raises
    ------
    ValueError
        If the required ``pivot_low`` / ``pivot_high`` column is absent.
    """
    pivot_col = "pivot_low" if kind == "low" else "pivot_high"
    price_col = "low" if kind == "low" else "high"

    if pivot_col not in df.columns:
        raise ValueError(
            f"Column '{pivot_col}' not found. Run find_pivots() first."
        )

    pivot_positions = np.nonzero(df[pivot_col].to_numpy())[0]  # integer iloc positions

    if len(pivot_positions) == 0:
        return None

    # Consider only the most recent `lookback_pivots` candidates.
    candidates = pivot_positions[-lookback_pivots:]
    pivot_prices = df[price_col].to_numpy(dtype=np.float64)[candidates]
    closes = df["close"].to_numpy(dtype=np.float64)
    total_bars = len(df)

    # Small O(K) loop — K ≤ lookback_pivots (≤ 5 by default).
    # Same pattern as the O(K) cluster loop in zones.py.
    magnitudes = np.empty(len(candidates), dtype=np.float64)
    for local_i, pos in enumerate(candidates):
        end = min(int(pos) + prominence_horizon + 1, total_bars)
        future_closes = closes[int(pos) : end]
        if kind == "low":
            magnitudes[local_i] = np.max(future_closes) - pivot_prices[local_i]
        else:
            magnitudes[local_i] = pivot_prices[local_i] - np.min(future_closes)

    return int(candidates[int(np.argmax(magnitudes))])


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_price_cols(df: pd.DataFrame) -> None:
    required = {"ts", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty.")

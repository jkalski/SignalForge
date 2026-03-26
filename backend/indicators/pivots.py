"""
Pivot detection module.

Identifies swing high and swing low pivots in OHLCV candle data using
vectorized methods. No Python-level loops over rows.

Primary method: scipy.signal.argrelextrema (O(n) in C)
Fallback method: numpy.lib.stride_tricks.sliding_window_view

A pivot_high at index i means df['high'][i] is the strict maximum
within the window [i - order, i + order].

A pivot_low at index i means df['low'][i] is the strict minimum
within the window [i - order, i + order].

Edge rows within `order` bars of either end cannot be pivots and are
set to False.

Adaptive order
--------------
``adaptive_pivot_order()`` computes the lookback window dynamically from
the current ATR relative to price.  In high-volatility regimes the window
widens so only significant structural pivots qualify; in calm regimes it
tightens to capture smaller but still valid swings.

Formula: max(5, int(atr_period * volatility_scalar))
where volatility_scalar = (atr / price) / base_atr_pct, capped at 2.0
to prevent unreasonably wide windows on extremely volatile instruments.
"""

import math

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema as _argrelextrema
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def adaptive_pivot_order(
    atr: float,
    price: float,
    atr_period: int = 14,
    base_atr_pct: float = 0.005,
) -> int:
    """
    Compute a pivot lookback order that adapts to current volatility.

    Uses: ``max(5, int(atr_period * volatility_scalar))``
    where ``volatility_scalar = (atr / price) / base_atr_pct``, capped at 2.0.

    Parameters
    ----------
    atr : float
        Current ATR14 of the symbol.
    price : float
        Current close price (> 0).
    atr_period : int, default 14
        ATR lookback period — used as the base multiplier.
    base_atr_pct : float, default 0.005
        ATR / price ratio considered "normal" volatility (0.5 %).
        Instruments with ATR % above this get a wider pivot window.

    Returns
    -------
    int
        Pivot order in [5, max(5, atr_period * 2)].
    """
    if price <= 0 or not math.isfinite(atr) or atr <= 0:
        return max(5, atr_period // 2)
    volatility_scalar = min((atr / price) / base_atr_pct, 2.0)
    return max(5, int(atr_period * volatility_scalar))


def find_pivots(df: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    """
    Detect swing high and swing low pivots in OHLCV candle data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ts, open, high, low, close, volume.
        Rows must be in chronological order.
    order : int, default 3
        Half-window size. A pivot requires the high/low to be the
        extremum across `order` bars to the left and `order` bars to
        the right (2 * order + 1 total window).

    Returns
    -------
    pd.DataFrame
        Input dataframe with two new boolean columns appended:
        - pivot_high : True where high[i] is the local maximum
        - pivot_low  : True where low[i] is the local minimum

    Raises
    ------
    ValueError
        If required columns are missing or `order` is less than 1.
    """
    required = {"high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    out = df.copy()

    if _SCIPY_AVAILABLE:
        out["pivot_high"], out["pivot_low"] = _pivots_scipy(out, order)
    else:
        out["pivot_high"], out["pivot_low"] = _pivots_numpy(out, order)

    return out


# ---------------------------------------------------------------------------
# Primary: scipy.signal.argrelextrema
# ---------------------------------------------------------------------------

def _pivots_scipy(df: pd.DataFrame, order: int):
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    n = len(highs)

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    high_idx = _argrelextrema(highs, np.greater, order=order)[0]
    low_idx = _argrelextrema(lows, np.less, order=order)[0]

    pivot_high[high_idx] = True
    pivot_low[low_idx] = True

    return pivot_high, pivot_low


# ---------------------------------------------------------------------------
# Fallback: numpy sliding_window_view
# ---------------------------------------------------------------------------

def _pivots_numpy(df: pd.DataFrame, order: int):
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    n = len(highs)
    w = 2 * order + 1

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    if n < w:
        # Not enough bars to form a single pivot window.
        return pivot_high, pivot_low

    # Shape: (n - w + 1, w)  — zero-copy view into the original array.
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, w)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, w)

    center = order  # column index of the center bar within each window

    center_high = high_windows[:, center]
    center_low = low_windows[:, center]

    # Split each window into left and right halves (excluding center).
    # This matches scipy's argrelextrema(np.greater) exactly:
    # pivot iff center is strictly greater/less than every surrounding bar.
    left_high_max = high_windows[:, :center].max(axis=1)
    right_high_max = high_windows[:, center + 1:].max(axis=1)
    left_low_min = low_windows[:, :center].min(axis=1)
    right_low_min = low_windows[:, center + 1:].min(axis=1)

    is_pivot_high = (center_high > left_high_max) & (center_high > right_high_max)
    is_pivot_low = (center_low < left_low_min) & (center_low < right_low_min)

    # Map window results back to original array indices.
    # Window i has its center at original index i + order.
    center_indices = np.arange(order, n - order)

    pivot_high[center_indices[is_pivot_high]] = True
    pivot_low[center_indices[is_pivot_low]] = True

    return pivot_high, pivot_low

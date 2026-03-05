"""
EMA trend filter.

Computes EMA 20 and EMA 50 and classifies each bar as bullish or bearish
relative to the slower EMA.  EMAs are used only as a directional filter —
crossovers produce no signals here.  Signal generation belongs exclusively
to the structure pipeline (zones, breakouts, bounces).

Usage
-----
After zone detection, pass the enriched DataFrame through add_trend_filter()
and use trend_bull / trend_bear to gate setups:

    df = add_trend_filter(df)
    long_setups  = zones where df['trend_bull'].iloc[-1] is True
    short_setups = zones where df['trend_bear'].iloc[-1] is True
"""

import pandas as pd


def add_trend_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ema20, ema50, trend_bull, and trend_bear columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column in chronological order.

    Returns
    -------
    pd.DataFrame
        Input dataframe with four new columns:
        - ema20      : 20-period EMA of close (float)
        - ema50      : 50-period EMA of close (float)
        - trend_bull : True where close > ema50 (bullish bias)
        - trend_bear : True where close < ema50 (bearish bias)

        trend_bull and trend_bear are mutually exclusive.
        Both are False when close == ema50 (exact equality, rare in practice)
        and during the EMA warmup period where ema50 may not be meaningful.

    Raises
    ------
    ValueError
        If the 'close' column is missing.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame is missing required column: 'close'")

    out = df.copy()
    close = out["close"].astype(float)

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    out["trend_bull"] = close > out["ema50"]
    out["trend_bear"] = close < out["ema50"]

    return out

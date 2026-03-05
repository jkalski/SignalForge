"""
Volume spike detection.

Adds a 20-period simple moving average of volume and a boolean spike flag
to a candle DataFrame.  Intended as a breakout confirmation input.
"""

import pandas as pd

_VOL_WINDOW = 20
_SPIKE_MULTIPLIER = 1.8


def add_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append vol_sma and vol_spike columns to a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'volume' column.  Rows must be in chronological order.

    Returns
    -------
    pd.DataFrame
        Same dataframe with two new columns:
        - vol_sma   : 20-period simple moving average of volume (float)
        - vol_spike : True where volume > vol_sma * 1.8 (bool)
                      False for the first 19 rows where vol_sma is NaN.

    Raises
    ------
    ValueError
        If the 'volume' column is missing.
    """
    if "volume" not in df.columns:
        raise ValueError("DataFrame is missing required column: 'volume'")

    out = df.copy()

    out["vol_sma"] = (
        out["volume"]
        .rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW)
        .mean()
    )

    # Rows where vol_sma is NaN (warmup period) are conservatively False.
    out["vol_spike"] = out["volume"] > out["vol_sma"] * _SPIKE_MULTIPLIER

    return out

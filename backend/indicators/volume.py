"""
Volume spike detection with time-of-day normalization (RVOL).

Standard vol/SMA ratio over-triggers at open/close because those bars
always have elevated volume.  RVOL fixes this by comparing each bar only
to the historical average for that same time-of-day slot.

Two signals are produced:
  vol_ratio  — raw volume / 20-bar rolling SMA (backward-compat)
  vol_spike  — True when RVOL >= 1.5x AND volume is rising vs prior bar
               (prevents late entries on exhaustion spikes)
  rvol       — time-of-day normalized ratio (vol / avg_for_this_time_slot)
"""

import pandas as pd

_VOL_WINDOW       = 20     # rolling SMA window (raw ratio, backward compat)
_RVOL_MIN_SAMPLES = 3      # minimum historical samples needed to use RVOL
_RVOL_SPIKE       = 1.5    # RVOL threshold for vol_spike gate
_RISING_LOOKBACK  = 1      # bars back to check volume is rising


def add_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append vol_sma, vol_ratio, rvol, and vol_spike columns to a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'volume' and 'ts' columns in chronological order.

    Returns
    -------
    pd.DataFrame with additional columns:
        vol_sma   : 20-bar rolling mean volume
        vol_ratio : volume / vol_sma  (raw, all-bars average)
        rvol      : volume / avg_volume_for_this_time_slot (time-of-day normalized)
        vol_spike : True when rvol >= 1.5x AND volume > prior bar volume
    """
    if "volume" not in df.columns:
        raise ValueError("DataFrame is missing required column: 'volume'")
    if "ts" not in df.columns:
        raise ValueError("DataFrame is missing required column: 'ts'")

    out = df.copy()

    # ── Raw rolling SMA (backward compat) ────────────────────────────────────
    out["vol_sma"] = (
        out["volume"]
        .rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW)
        .mean()
    )
    out["vol_ratio"] = out["volume"] / out["vol_sma"]

    # ── Time-of-day slot ─────────────────────────────────────────────────────
    # Use (hour, minute) as the slot key so 9:30 bars are only compared to
    # other 9:30 bars, not to midday bars.
    ts = pd.to_datetime(out["ts"])
    out["_tod"] = ts.dt.hour * 60 + ts.dt.minute  # minutes since midnight

    # For each bar, compute the mean volume of all *prior* bars in the same
    # time slot (expanding, not rolling — we want every historical sample).
    rvol_vals = []
    slot_history: dict[int, list[float]] = {}

    for _, row in out.iterrows():
        slot = int(row["_tod"])
        vol  = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
        hist = slot_history.get(slot, [])

        if len(hist) >= _RVOL_MIN_SAMPLES:
            avg = sum(hist) / len(hist)
            rvol = vol / avg if avg > 0 else 1.0
        else:
            # Fall back to raw vol_ratio during warmup
            raw = row["vol_ratio"]
            rvol = float(raw) if pd.notna(raw) else 1.0

        rvol_vals.append(round(rvol, 4))
        hist.append(vol)
        slot_history[slot] = hist

    out["rvol"] = rvol_vals

    # ── Vol spike: RVOL threshold AND volume rising ───────────────────────────
    vol_rising = out["volume"] > out["volume"].shift(_RISING_LOOKBACK)
    out["vol_spike"] = (out["rvol"] >= _RVOL_SPIKE) & vol_rising

    out.drop(columns=["_tod"], inplace=True)

    return out

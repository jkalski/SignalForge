import math
import pandas as pd
from typing import Any, Dict, List


def _clean(val: Any) -> Any:
    """Convert float NaN and pandas NaT to None for JSON-safe output."""
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def compute_features(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of candle dicts sorted ascending by ts and returns the same
    rows augmented with: returns, ema_20, ema_50, rsi_14, atr_14.
    Rows where indicators haven't warmed up yet will have None values.
    """
    if not candles:
        return []

    df = pd.DataFrame(candles)
    df = df.sort_values("ts").reset_index(drop=True)

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # Percent returns
    df["returns"] = close.pct_change().round(6)

    # Exponential moving averages
    df["ema_20"] = close.ewm(span=20, adjust=False).mean().round(4)
    df["ema_50"] = close.ewm(span=50, adjust=False).mean().round(4)

    # RSI 14 (Wilder's smoothing via ewm alpha=1/14)
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs       = avg_gain / avg_loss
    df["rsi_14"] = (100 - (100 / (1 + rs))).round(2)

    # ATR 14 (True Range, Wilder's smoothing)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean().round(4)

    # Walk each record and replace NaN/NaT with None (float NaN is invalid JSON)
    return [
        {k: _clean(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

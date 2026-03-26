"""backend/ml/features.py

Feature encoding pipeline — converts raw setup dicts and parquet rows into
a clean numeric matrix that XGBoost can train on.

Responsibilities
----------------
- Encode categorical columns (event_type, direction, htf_bias, trend,
  signal_status) as integer codes using a fixed vocabulary
- Fill NaN values with sensible defaults (median or -1 sentinel)
- Return a consistent feature column list so training and inference
  use the exact same input shape

The vocabulary mappings are module-level constants so the predictor
can encode a live signal dict using the exact same logic as the trainer,
without needing a fitted sklearn encoder or extra files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Categorical vocabularies — fixed at definition time
# All unknown values map to 0 ("unknown" / "other")
# ---------------------------------------------------------------------------

EVENT_TYPE_MAP: Dict[str, int] = {
    "unknown":                  0,
    "breakout_up":              1,
    "breakdown_down":           2,
    "bounce_up":                3,
    "reject_down":              4,
    "sweep_up":                 5,
    "sweep_down":               6,
    "bos_up":                   7,
    "bos_down":                 8,
    "choch_up":                 9,
    "choch_down":               10,
    "fvg_long":                 11,
    "fvg_short":                12,
    "vwap_reclaim_long":        13,
    "vwap_reclaim_short":       14,
    "orb_long":                 15,
    "orb_short":                16,
    "gap_fade_long":            17,
    "gap_fade_short":           18,
    "gap_go_long":              19,
    "gap_go_short":             20,
    "ath_breakout_long":        21,
    "inside_bar_long":          22,
    "inside_bar_short":         23,
    "double_inside_bar_long":   24,
    "double_inside_bar_short":  25,
    "outside_bar_long":         26,
    "outside_bar_short":        27,
}

DIRECTION_MAP: Dict[str, int] = {
    "unknown": 0,
    "long":    1,
    "short":   2,
}

HTF_BIAS_MAP: Dict[str, int] = {
    "unknown":  0,
    "neutral":  1,
    "bullish":  2,
    "bearish":  3,
}

TREND_MAP: Dict[str, int] = {
    "unknown": 0,
    "neutral": 1,
    "bull":    2,
    "bear":    3,
}

SIGNAL_STATUS_MAP: Dict[str, int] = {
    "unknown":   0,
    "watchlist": 1,
    "active":    2,
}

# ---------------------------------------------------------------------------
# Ordered feature columns — must match between trainer and predictor
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    # Encoded categoricals
    "event_type_enc",
    "direction_enc",
    "htf_bias_enc",
    "trend_enc",
    "signal_status_enc",
    # Zone
    "zone_touches",
    "zones_ltf_count",
    "zones_htf_count",
    "distance_pct",
    # Indicators
    "vol_ratio",
    "vol_spike",
    "ema_confirms",
    "atr_14",
    # HTF / MTF
    "mtf_aligned",
    "near_htf_zone",
    # VWAP
    "vwap_session_dist_pct",
    "vwap_anchored_dist_pct",
    # Scoring
    "confluence_score",
    "score_zone",
    "score_volume",
    "score_event",
    "score_ema",
    "score_vwap",
    "score_mtf",
    "score_divergence",
]

# Numeric columns that may contain NaN — filled with median during training,
# then with the training median during inference.
_NULLABLE_COLS = [
    "vol_ratio",
    "vwap_session_dist_pct",
    "vwap_anchored_dist_pct",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_dataframe(
    df: pd.DataFrame,
    medians: Optional[Dict[str, float]] = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Encode a raw feature DataFrame (from the parquet) into model-ready form.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows from build_training_data() or equivalent.
    medians : dict or None
        Pre-computed medians for nullable columns.  Pass None during training
        (medians are computed from df and returned).  Pass the training medians
        during inference so the same fill values are used.

    Returns
    -------
    encoded : pd.DataFrame
        DataFrame with FEATURE_COLS columns, no NaN values.
    medians : dict
        Median values used for NaN-filling (same dict passed in, or freshly
        computed if None was passed).
    """
    out = pd.DataFrame(index=df.index)

    # ── Categorical encoding ──────────────────────────────────────────────
    out["event_type_enc"]    = df["event_type"].map(EVENT_TYPE_MAP).fillna(0).astype(int)
    out["direction_enc"]     = df["direction"].map(DIRECTION_MAP).fillna(0).astype(int)
    out["htf_bias_enc"]      = df["htf_bias"].map(HTF_BIAS_MAP).fillna(0).astype(int)
    out["trend_enc"]         = df["trend"].map(TREND_MAP).fillna(0).astype(int)
    out["signal_status_enc"] = df["signal_status"].map(SIGNAL_STATUS_MAP).fillna(0).astype(int)

    # ── Numeric columns — copy as-is ─────────────────────────────────────
    numeric_cols = [c for c in FEATURE_COLS if c not in out.columns]
    for col in numeric_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan

    # ── NaN filling ───────────────────────────────────────────────────────
    if medians is None:
        medians = {
            col: float(out[col].median())
            for col in _NULLABLE_COLS
            if col in out.columns
        }

    for col, med in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(med if np.isfinite(med) else 0.0)

    # Fill any remaining NaN with 0
    out = out.fillna(0)

    return out[FEATURE_COLS], medians


def encode_signal(
    signal: Dict[str, Any],
    medians: Dict[str, float],
) -> np.ndarray:
    """
    Encode a single live signal dict into a 1-row feature array for inference.

    Parameters
    ----------
    signal : dict
        Output from run_structure_pipeline() or equivalent.
        Must contain the same keys as the parquet feature columns.
    medians : dict
        Training medians from the saved model metadata — used to fill
        NaN values consistently with training.

    Returns
    -------
    np.ndarray, shape (1, len(FEATURE_COLS))
    """
    row: Dict[str, Any] = {}

    # Categoricals
    row["event_type_enc"]    = EVENT_TYPE_MAP.get(signal.get("event_type", ""), 0)
    row["direction_enc"]     = DIRECTION_MAP.get(signal.get("direction", ""), 0)
    row["htf_bias_enc"]      = HTF_BIAS_MAP.get(signal.get("htf_bias", "neutral") or "neutral", 1)
    row["trend_enc"]         = TREND_MAP.get(signal.get("trend", "neutral") or "neutral", 1)
    row["signal_status_enc"] = SIGNAL_STATUS_MAP.get(signal.get("signal_status", "watchlist") or "watchlist", 1)

    # Zone
    row["zone_touches"]    = int(signal.get("zone_touches") or 0)
    row["zones_ltf_count"] = int(signal.get("zones_ltf_count") or 0)
    row["zones_htf_count"] = int(signal.get("zones_htf_count") or 0)
    row["distance_pct"]    = float(signal.get("distance_pct") or 0.0)

    # Indicators
    vr = signal.get("vol_ratio")
    row["vol_ratio"]    = float(vr) if vr is not None else medians.get("vol_ratio", 1.0)
    row["vol_spike"]    = int(bool(signal.get("vol_spike", False)))
    row["ema_confirms"] = int(bool(signal.get("ema_confirms", False)))
    row["atr_14"]       = float(signal.get("atr_14") or signal.get("atr14") or 0.0)

    # HTF / MTF
    row["mtf_aligned"]   = int(bool(signal.get("mtf_aligned", False)))
    row["near_htf_zone"] = int(bool(signal.get("near_htf_zone", False)))

    # VWAP
    def _vwap(key: str) -> float:
        v = signal.get(key)
        return float(v) if v is not None else medians.get(key, 0.0)

    row["vwap_session_dist_pct"]  = _vwap("vwap_session_dist_pct")
    row["vwap_anchored_dist_pct"] = _vwap("vwap_anchored_dist_pct")

    # Scoring
    row["confluence_score"] = int(signal.get("confluence_score") or 0)

    reasons = signal.get("confluence_reasons") or {}
    def _sub(dim: str) -> int:
        d = reasons.get(dim)
        return int(d.get("score", 0)) if isinstance(d, dict) else 0

    row["score_zone"]       = _sub("zone")
    row["score_volume"]     = _sub("volume")
    row["score_event"]      = _sub("event")
    row["score_ema"]        = _sub("ema")
    row["score_vwap"]       = _sub("vwap")
    row["score_mtf"]        = _sub("mtf")
    row["score_divergence"] = _sub("divergence")

    return np.array([[row[col] for col in FEATURE_COLS]], dtype=np.float32)

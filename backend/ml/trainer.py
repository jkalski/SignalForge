"""backend/ml/trainer.py

XGBoost model trainer.

Loads the labeled feature parquet, encodes it, trains a gradient boosting
classifier to predict setup win probability, and saves the model + metadata
to disk.

The saved artifact (model.pkl) contains everything needed for inference:
  - trained XGBoost model
  - training medians (for NaN-filling at inference time)
  - feature column list
  - training metadata (date, rows, win_rate, feature importances)

Usage
-----
    python -m backend.ml.trainer                        # default paths
    python -m backend.ml.trainer --data training_data/features.parquet
    python -m backend.ml.trainer --out training_data/model.pkl
    python -m backend.ml.trainer --min-score 40         # filter low-quality rows
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backend.ml.features import FEATURE_COLS, encode_dataframe

logger = logging.getLogger(__name__)

_DEFAULT_DATA = Path("training_data") / "features.parquet"
_DEFAULT_OUT  = Path("training_data") / "model.pkl"
_DEFAULT_META = Path("training_data") / "model_meta.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    data_path:  str | Path = _DEFAULT_DATA,
    out_path:   str | Path = _DEFAULT_OUT,
    min_score:  int        = 0,
    test_size:  float      = 0.2,
    random_seed: int       = 42,
) -> Dict[str, Any]:
    """
    Train an XGBoost win-probability model on the backtest feature parquet.

    Parameters
    ----------
    data_path : path
        Parquet file produced by backend.ml.data.build_training_data().
    out_path : path
        Where to save the model pickle.
    min_score : int
        Drop rows with confluence_score below this threshold before training.
        0 = keep everything.
    test_size : float
        Fraction of data held out for evaluation (time-ordered split).
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    dict
        Training metadata: accuracy, auc, feature importances, row counts.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Run: pip install xgboost scikit-learn"
        ) from e

    t0 = time.monotonic()
    data_path = Path(data_path)
    out_path  = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading training data: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── 2. Filter ─────────────────────────────────────────────────────────
    if min_score > 0:
        before = len(df)
        df = df[df["confluence_score"] >= min_score]
        logger.info("Score filter (>=%d): %d → %d rows", min_score, before, len(df))

    # Only active signals — vol_spike + ema_confirms both passed
    before = len(df)
    df = df[df["signal_status"] == "active"]
    logger.info("Active filter: %d → %d rows", before, len(df))

    # Drop timeouts — only train on resolved outcomes (win/loss)
    before = len(df)
    df = df[df["outcome"].isin(["win", "loss"])]
    logger.info("Dropped timeouts: %d → %d rows", before, len(df))

    if len(df) < 100:
        raise ValueError(f"Too few training rows after filtering: {len(df)}")

    # ── 3. Encode features ────────────────────────────────────────────────
    X, medians = encode_dataframe(df)
    y = df["win"].astype(int).values

    win_rate = float(y.mean())
    logger.info("Win rate: %.1f%% | pos=%d neg=%d", win_rate * 100, y.sum(), (1 - y).sum())

    # ── 4. Time-ordered train/test split ──────────────────────────────────
    # Sort by ts so test set is always the most recent data — avoids leakage.
    if "ts" in df.columns:
        df_sorted = df.sort_values("ts")
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_idx = df_sorted.index[:split_idx]
        test_idx  = df_sorted.index[split_idx:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y[df.index.get_indexer(train_idx)], y[df.index.get_indexer(test_idx)]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ── 5. Class weight (handle imbalance) ────────────────────────────────
    neg_count  = int((y_train == 0).sum())
    pos_count  = int((y_train == 1).sum())
    scale_pos  = neg_count / pos_count if pos_count > 0 else 1.0

    # ── 6. Train XGBoost ─────────────────────────────────────────────────
    logger.info("Training XGBoost (scale_pos_weight=%.2f)...", scale_pos)
    model = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos,
        eval_metric      = "auc",
        random_state     = random_seed,
        n_jobs           = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc      = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))

    logger.info("Test AUC: %.4f | Accuracy: %.4f", auc, accuracy)

    # ── 8. Feature importances ────────────────────────────────────────────
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features:")
    for feat, imp in top:
        logger.info("  %-30s %.4f", feat, imp)

    # ── 9. Save model ─────────────────────────────────────────────────────
    artifact = {
        "model":        model,
        "medians":      medians,
        "feature_cols": FEATURE_COLS,
        "trained_at":   datetime.now(UTC).isoformat(),
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
        "win_rate":     win_rate,
        "auc":          auc,
        "accuracy":     accuracy,
        "importances":  importances,
    }

    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Model saved → %s", out_path)

    # ── 10. Save human-readable meta ─────────────────────────────────────
    meta = {k: v for k, v in artifact.items() if k != "model"}
    meta_path = out_path.parent / (out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Meta → %s", meta_path)

    elapsed = round(time.monotonic() - t0, 2)
    logger.info("Done in %.2fs", elapsed)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost win-probability model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      default=str(_DEFAULT_DATA), help="Input parquet path")
    p.add_argument("--out",       default=str(_DEFAULT_OUT),  help="Output model.pkl path")
    p.add_argument("--min-score", type=int, default=0,        help="Min confluence_score to include")
    p.add_argument("--test-size", type=float, default=0.2,    help="Test set fraction")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        meta = train(
            data_path = args.data,
            out_path  = args.out,
            min_score = args.min_score,
            test_size = args.test_size,
        )
    except Exception as e:
        print(f"[ml.trainer] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[ml.trainer] Model trained successfully")
    print(f"  train rows:  {meta['train_rows']:,}")
    print(f"  test rows:   {meta['test_rows']:,}")
    print(f"  win rate:    {meta['win_rate']:.1%}")
    print(f"  test AUC:    {meta['auc']:.4f}")
    print(f"  accuracy:    {meta['accuracy']:.4f}")
    print(f"\nTop 10 features by importance:")
    top = sorted(meta["importances"].items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in top:
        print(f"  {feat:<30} {imp:.4f}")

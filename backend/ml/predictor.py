"""backend/ml/predictor.py

Real-time signal scorer using the trained XGBoost model.

Loads model.pkl once at startup and keeps it in memory.  Every new signal
that comes through the pipeline gets scored in microseconds.

Usage (as a module)
-------------------
    from backend.ml.predictor import get_predictor

    predictor = get_predictor()          # loads model once, cached
    prob = predictor.score(signal_dict)  # returns float 0.0–1.0 or None

The predictor returns None (not 0.0) when the model file doesn't exist yet,
so the pipeline can fall back to the rule-based confluence_score gracefully
without any code changes.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.ml.features import FEATURE_COLS, encode_signal

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("training_data") / "model.pkl"


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class Predictor:
    """
    Thread-safe wrapper around a trained XGBoost model.

    Loaded once and reused across all pipeline calls.  Supports hot-reloading
    via reload() so the scheduler can retrain and swap the model without
    restarting the server.
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        self._path    = Path(model_path)
        self._lock    = threading.RLock()
        self._model   = None
        self._medians: Dict[str, float] = {}
        self._meta:    Dict[str, Any]   = {}
        self._loaded  = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.info(
                "Model not found at %s — predictor will return None until trained.",
                self._path,
            )
            return

        try:
            with open(self._path, "rb") as f:
                artifact = pickle.load(f)

            with self._lock:
                self._model   = artifact["model"]
                self._medians = artifact.get("medians", {})
                self._meta    = {k: v for k, v in artifact.items() if k != "model"}
                self._loaded  = True

            logger.info(
                "Model loaded | trained_at=%s | auc=%.4f | rows=%d",
                self._meta.get("trained_at", "?"),
                self._meta.get("auc", 0),
                self._meta.get("train_rows", 0),
            )
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", self._path, e)

    def reload(self) -> None:
        """Hot-reload the model from disk (call after retraining)."""
        logger.info("Reloading model from %s", self._path)
        self._loaded = False
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    def score(self, signal: Dict[str, Any]) -> Optional[float]:
        """
        Score a single signal dict and return win probability in [0, 1].

        Returns None if the model hasn't been trained yet, so callers can
        fall back to the rule-based confluence_score without any special casing.

        Parameters
        ----------
        signal : dict
            Pipeline output dict.  Must contain the same keys as the training
            feature set (event_type, direction, zone_touches, vol_ratio, …).

        Returns
        -------
        float in [0.0, 1.0] or None
        """
        if not self.is_ready:
            return None

        try:
            with self._lock:
                X = encode_signal(signal, self._medians)
                prob = float(self._model.predict_proba(X)[0, 1])
            return round(prob, 4)
        except Exception as e:
            logger.debug("Scoring failed: %s", e)
            return None

    def score_batch(self, signals: list[Dict[str, Any]]) -> list[Optional[float]]:
        """Score multiple signals at once — more efficient than calling score() in a loop."""
        if not self.is_ready or not signals:
            return [None] * len(signals)

        try:
            with self._lock:
                rows = np.vstack([encode_signal(s, self._medians) for s in signals])
                probs = self._model.predict_proba(rows)[:, 1]
            return [round(float(p), 4) for p in probs]
        except Exception as e:
            logger.debug("Batch scoring failed: %s", e)
            return [None] * len(signals)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_predictor: Optional[Predictor] = None
_predictor_lock = threading.Lock()


def get_predictor(model_path: str | Path = _DEFAULT_MODEL_PATH) -> Predictor:
    """
    Return the module-level Predictor singleton, creating it on first call.

    Thread-safe.  Safe to call from multiple pipeline worker threads.
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = Predictor(model_path)
    return _predictor


def reload_predictor() -> None:
    """Hot-reload the model after retraining.  Called by the scheduler."""
    global _predictor
    if _predictor is not None:
        _predictor.reload()
    else:
        get_predictor()

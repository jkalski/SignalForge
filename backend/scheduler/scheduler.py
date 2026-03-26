"""backend/scheduler/scheduler.py

Background scheduler for the trading pipeline.

Runs three recurring jobs in a daemon thread, sequentially, to respect
data dependencies:

  1. agent_job        — scans the universe, persists setups + signals   (every 1h)
  2. outcomes_job     — evaluates signal outcomes from candle history    (every 6h)
  3. probability_job  — aggregates outcomes into ProbabilityHistory      (every 24h)

The scheduler is started at FastAPI startup and stopped at shutdown via
start_scheduler() / stop_scheduler().  It never blocks the main thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job intervals
# ---------------------------------------------------------------------------

_AGENT_INTERVAL_S       = 3_600    # 1 hour
_OUTCOMES_INTERVAL_S    = 21_600   # 6 hours
_PROBABILITY_INTERVAL_S = 86_400   # 24 hours
_RETRAIN_INTERVAL_S     = 604_800  # 7 days

# ---------------------------------------------------------------------------
# Default pipeline parameters (can be overridden via start_scheduler kwargs)
# ---------------------------------------------------------------------------

_TIMEFRAME = "1h"
_SOURCE    = "yahoo"
_TOP_N     = 10
_LOOKBACK  = 200
_HORIZONS  = [6, 24, 72]


# ---------------------------------------------------------------------------
# Job functions
# ---------------------------------------------------------------------------


def _run_agent(timeframe: str, source: str, top_n: int, lookback: int) -> None:
    from backend.agent.runner import run_agent
    try:
        result = run_agent(
            timeframe = timeframe,
            source    = source,
            top_n     = top_n,
            lookback  = lookback,
        )
        log.info("[scheduler] agent done: %s", result)
    except Exception as e:
        log.error("[scheduler] agent failed: %s", e)


def _run_outcomes(timeframe: str, source: str, horizons: list) -> None:
    from backend.agent.evaluate_outcomes import evaluate_outcomes
    try:
        result = evaluate_outcomes(timeframe=timeframe, source=source, horizons=horizons)
        log.info("[scheduler] outcomes done: %s", result)
    except Exception as e:
        log.error("[scheduler] outcomes failed: %s", e)


def _run_probability(timeframe: str, horizons: list) -> None:
    from backend.probability.aggregator import aggregate_probabilities
    try:
        for horizon in horizons:
            result = aggregate_probabilities(timeframe=timeframe, horizon_bars=horizon)
            log.info("[scheduler] probability horizon=%d done: %s", horizon, result)
    except Exception as e:
        log.error("[scheduler] probability failed: %s", e)


def _run_retrain() -> None:
    from backend.ml.trainer import train
    from backend.ml.predictor import reload_predictor
    try:
        metrics = train()
        log.info("[scheduler] ml retrain done: %s", metrics)
        reload_predictor()
        log.info("[scheduler] predictor reloaded")
    except Exception as e:
        log.error("[scheduler] ml retrain failed: %s", e)


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------


class _Scheduler:
    def __init__(
        self,
        timeframe: str,
        source:    str,
        top_n:     int,
        lookback:  int,
        horizons:  list,
    ) -> None:
        self._timeframe = timeframe
        self._source    = source
        self._top_n     = top_n
        self._lookback  = lookback
        self._horizons  = horizons

        self._stop               = threading.Event()
        self._last_agent         = 0.0
        self._last_outcomes      = 0.0
        self._last_probability   = 0.0
        self._last_retrain       = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()

            if now - self._last_agent >= _AGENT_INTERVAL_S:
                _run_agent(self._timeframe, self._source, self._top_n, self._lookback)
                self._last_agent = time.monotonic()

            if now - self._last_outcomes >= _OUTCOMES_INTERVAL_S:
                _run_outcomes(self._timeframe, self._source, self._horizons)
                self._last_outcomes = time.monotonic()

            if now - self._last_probability >= _PROBABILITY_INTERVAL_S:
                _run_probability(self._timeframe, self._horizons)
                self._last_probability = time.monotonic()

            if now - self._last_retrain >= _RETRAIN_INTERVAL_S:
                _run_retrain()
                self._last_retrain = time.monotonic()

            # Wake up every 60 s to check if any job is due.
            self._stop.wait(60)

    def start(self) -> None:
        t = threading.Thread(target=self._loop, daemon=True, name="pipeline-scheduler")
        t.start()
        log.info(
            "[scheduler] started — agent=1h, outcomes=6h, probability=24h, retrain=7d "
            "(timeframe=%s, source=%s)",
            self._timeframe, self._source,
        )

    def stop(self) -> None:
        self._stop.set()
        log.info("[scheduler] stopped")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[_Scheduler] = None


def start_scheduler(
    timeframe: str  = _TIMEFRAME,
    source:    str  = _SOURCE,
    top_n:     int  = _TOP_N,
    lookback:  int  = _LOOKBACK,
    horizons:  list = None,
) -> None:
    """Start the background scheduler.  Call once at app startup."""
    global _scheduler
    if _scheduler is not None:
        return  # already running
    _scheduler = _Scheduler(
        timeframe = timeframe,
        source    = source,
        top_n     = top_n,
        lookback  = lookback,
        horizons  = horizons or _HORIZONS,
    )
    _scheduler.start()


def stop_scheduler() -> None:
    """Stop the background scheduler.  Call at app shutdown."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None

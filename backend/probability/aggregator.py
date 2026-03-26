"""backend/probability/aggregator.py

Aggregates SignalOutcome rows into ProbabilityHistory win-rate statistics.

Groups outcomes by (setup_type, timeframe) within a rolling lookback window
and computes: sample_count, win_rate, avg_r_won, avg_r_lost, expected_r.

Also exposes get_win_rate() — a fast single-row lookup used by the agent
runner to populate Signal.prob_success at signal creation time.

CLI usage:
    python -m backend.probability.aggregator
    python -m backend.probability.aggregator --timeframe 1h --horizon 24 --lookback-days 180
    python -m backend.probability.aggregator --min-samples 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from backend.db.models import ProbabilityHistory, Signal, SignalOutcome
from backend.db.session import SessionLocal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HORIZON     = 24    # bars
_DEFAULT_LOOKBACK    = 180   # calendar days
_MIN_SAMPLES         = 10    # skip setup types with fewer outcomes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_probabilities(
    timeframe:     str = "1h",
    horizon_bars:  int = _DEFAULT_HORIZON,
    lookback_days: int = _DEFAULT_LOOKBACK,
    min_samples:   int = _MIN_SAMPLES,
) -> dict:
    """
    Compute win-rate statistics per (setup_type, timeframe) from SignalOutcome
    rows and insert new ProbabilityHistory records.

    Each call inserts fresh rows (append-only); callers should use
    get_win_rate() which orders by computed_at desc to get the latest.

    Returns a summary dict with counts.
    """
    db: Session = SessionLocal()
    now          = datetime.now(UTC)
    window_start = now - timedelta(days=lookback_days)

    inserted = 0
    skipped  = 0
    groups   = 0

    try:
        # Fetch outcomes for this timeframe / horizon within the window.
        outcomes = (
            db.query(SignalOutcome)
            .filter(
                SignalOutcome.timeframe    == timeframe,
                SignalOutcome.horizon_bars == horizon_bars,
                SignalOutcome.signal_ts    >= window_start,
                SignalOutcome.return_pct   != None,
            )
            .all()
        )

        if not outcomes:
            return {
                "timeframe":    timeframe,
                "horizon_bars": horizon_bars,
                "groups":       0,
                "inserted":     0,
                "skipped":      0,
            }

        # Resolve setup_type for each outcome via its parent Signal.
        signal_ids = list({o.signal_id for o in outcomes})
        signals    = (
            db.query(Signal.id, Signal.setup_type)
            .filter(Signal.id.in_(signal_ids))
            .all()
        )
        setup_type_map: Dict[str, str] = {s.id: s.setup_type for s in signals}

        # Group outcomes by setup_type.
        by_setup: Dict[str, List[SignalOutcome]] = {}
        for o in outcomes:
            st = setup_type_map.get(o.signal_id)
            if not st:
                continue
            by_setup.setdefault(st, []).append(o)

        for setup_type, group in by_setup.items():
            groups += 1
            if len(group) < min_samples:
                skipped += 1
                continue

            returns = [float(o.return_pct) for o in group]
            wins    = [r for r in returns if r > 0]
            losses  = [r for r in returns if r <= 0]

            win_rate   = round(len(wins) / len(returns), 4)
            avg_r_won  = round(sum(wins)   / len(wins),   4) if wins   else None
            avg_r_lost = round(sum(losses) / len(losses), 4) if losses else None

            if avg_r_won is not None and avg_r_lost is not None:
                expected_r = round(win_rate * avg_r_won + (1 - win_rate) * avg_r_lost, 4)
            else:
                expected_r = None

            db.add(ProbabilityHistory(
                setup_type        = setup_type,
                timeframe         = timeframe,
                symbol            = None,       # global aggregate across all symbols
                window_start      = window_start,
                window_end        = now,
                sample_count      = len(returns),
                win_count         = len(wins),
                win_rate          = win_rate,
                avg_r_won         = avg_r_won,
                avg_r_lost        = avg_r_lost,
                expected_r        = expected_r,
                calibration_score = None,
                computed_at       = now,
            ))
            inserted += 1

        if inserted:
            db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "timeframe":    timeframe,
        "horizon_bars": horizon_bars,
        "groups":       groups,
        "inserted":     inserted,
        "skipped":      skipped,
    }


def get_win_rate(
    db: Session,
    setup_type: str,
    timeframe: str,
) -> Optional[float]:
    """
    Return the most recently computed win rate for a (setup_type, timeframe)
    pair, or None if no history exists yet.

    Designed to be called inside the agent runner at signal creation time.
    """
    row = (
        db.query(ProbabilityHistory)
        .filter(
            ProbabilityHistory.setup_type == setup_type,
            ProbabilityHistory.timeframe  == timeframe,
            ProbabilityHistory.symbol     == None,
        )
        .order_by(ProbabilityHistory.computed_at.desc())
        .first()
    )
    return float(row.win_rate) if row and row.win_rate is not None else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate signal outcomes into probability history")
    p.add_argument("--timeframe",    default="1h",  help="Bar timeframe (default: 1h)")
    p.add_argument("--horizon",      type=int, default=_DEFAULT_HORIZON,
                   help=f"Horizon bar count (default: {_DEFAULT_HORIZON})")
    p.add_argument("--lookback-days", type=int, default=_DEFAULT_LOOKBACK,
                   help=f"Days of outcome history to include (default: {_DEFAULT_LOOKBACK})")
    p.add_argument("--min-samples",  type=int, default=_MIN_SAMPLES,
                   help=f"Min outcomes required to write a record (default: {_MIN_SAMPLES})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        result = aggregate_probabilities(
            timeframe     = args.timeframe,
            horizon_bars  = args.horizon,
            lookback_days = args.lookback_days,
            min_samples   = args.min_samples,
        )
    except Exception as e:
        print(f"[aggregator] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[aggregator] timeframe:    {result['timeframe']}")
    print(f"[aggregator] horizon_bars: {result['horizon_bars']}")
    print(f"[aggregator] groups found: {result['groups']}")
    print(f"[aggregator] inserted:     {result['inserted']}")
    print(f"[aggregator] skipped:      {result['skipped']}  (< {args.min_samples} samples)")

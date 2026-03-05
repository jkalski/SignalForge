"""backend/agent/evaluate_outcomes.py

Evaluates historical outcomes for stored signals by looking up candles that
followed each signal and computing direction-aware return / MFE / MAE.

Usage:
    python -m backend.agent.evaluate_outcomes
    python -m backend.agent.evaluate_outcomes --timeframe 1h --source yahoo --horizons 6,24,72
    python -m backend.agent.evaluate_outcomes --limit 500
"""

import argparse
import sys
from datetime import datetime, UTC
from typing import List, Optional

from sqlalchemy import and_, not_, exists
from sqlalchemy.orm import Session

from backend.db.models import Candle, Signal, SignalOutcome
from backend.db.session import SessionLocal


def _evaluate_signal(
    db: Session,
    signal: Signal,
    horizon_bars: int,
    source: Optional[str],
) -> Optional[SignalOutcome]:
    """
    Build a SignalOutcome for one (signal, horizon) pair.
    Returns None if there are not enough candles after signal_ts yet.
    """
    outcome_id = f"{signal.id}_{horizon_bars}"

    # Idempotency check
    if db.get(SignalOutcome, outcome_id) is not None:
        return None

    # Fetch the next horizon_bars candles strictly after signal_ts
    q = (
        db.query(Candle)
        .filter(
            Candle.symbol    == signal.symbol,
            Candle.timeframe == signal.timeframe,
            Candle.ts        > signal.created_at,
        )
    )
    if source:
        q = q.filter(Candle.source == source)

    candles = q.order_by(Candle.ts.asc()).limit(horizon_bars).all()

    if len(candles) < horizon_bars:
        # Horizon not yet complete — skip, will be picked up on next run
        return None

    entry = float(signal.entry_price) if signal.entry_price is not None else None
    if entry is None or entry == 0:
        return None

    direction = signal.direction  # "long" or "short"

    exit_candle = candles[-1]
    exit_price  = float(exit_candle.close)
    exit_ts     = exit_candle.ts

    highs = [float(c.high) for c in candles]
    lows  = [float(c.low)  for c in candles]

    if direction == "long":
        return_pct        = (exit_price - entry) / entry * 100
        max_favorable_pct = (max(highs) - entry) / entry * 100   # MFE
        max_adverse_pct   = (min(lows)  - entry) / entry * 100   # MAE (negative)
    else:  # short
        return_pct        = (entry - exit_price) / entry * 100
        max_favorable_pct = (entry - min(lows))  / entry * 100   # MFE
        max_adverse_pct   = (entry - max(highs)) / entry * 100   # MAE (negative)

    return SignalOutcome(
        id                = outcome_id,
        signal_id         = signal.id,
        symbol            = signal.symbol,
        timeframe         = signal.timeframe,
        source            = source,
        signal_ts         = signal.created_at,
        horizon_bars      = horizon_bars,
        entry_price       = round(entry, 4),
        exit_ts           = exit_ts,
        exit_price        = round(exit_price, 4),
        return_pct        = round(return_pct, 6),
        max_favorable_pct = round(max_favorable_pct, 6),
        max_adverse_pct   = round(max_adverse_pct, 6),
        created_at        = datetime.now(UTC),
    )


def evaluate_outcomes(
    timeframe: str = "1h",
    source: Optional[str] = "yahoo",
    horizons: List[int] = None,
    limit: int = 1000,
) -> dict:
    """
    Find signals that are missing outcome rows for any requested horizon,
    evaluate them, and insert the results.

    Returns a summary dict with evaluated/inserted/skipped counts.
    """
    if horizons is None:
        horizons = [6, 24, 72]

    db: Session = SessionLocal()
    evaluated = 0
    inserted  = 0
    skipped   = 0

    try:
        for horizon in horizons:
            # Find signals that don't yet have an outcome for this horizon.
            # Use a NOT EXISTS subquery so we only fetch what's missing.
            missing_signals = (
                db.query(Signal)
                .filter(
                    Signal.timeframe == timeframe,
                    not_(
                        exists().where(
                            and_(
                                SignalOutcome.signal_id    == Signal.id,
                                SignalOutcome.horizon_bars == horizon,
                            )
                        )
                    ),
                )
                .order_by(Signal.created_at.asc())
                .limit(limit)
                .all()
            )

            for signal in missing_signals:
                evaluated += 1
                outcome = _evaluate_signal(db, signal, horizon, source)
                if outcome is None:
                    skipped += 1
                    continue
                db.add(outcome)
                inserted += 1

            if inserted:
                db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "horizons":  horizons,
        "evaluated": evaluated,
        "inserted":  inserted,
        "skipped":   skipped,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate signal outcomes against candle history")
    p.add_argument("--timeframe", default="1h",    help="Bar timeframe (default: 1h)")
    p.add_argument("--source",    default="yahoo", help="Candle source filter (default: yahoo)")
    p.add_argument(
        "--horizons",
        default="6,24,72",
        help="Comma-separated horizon bar counts (default: 6,24,72)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max signals to evaluate per horizon per run (default: 1000)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    try:
        result = evaluate_outcomes(
            timeframe = args.timeframe,
            source    = args.source,
            horizons  = horizons,
            limit     = args.limit,
        )
    except Exception as e:
        print(f"[evaluate_outcomes] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluate_outcomes] horizons:  {result['horizons']}")
    print(f"[evaluate_outcomes] evaluated: {result['evaluated']}")
    print(f"[evaluate_outcomes] inserted:  {result['inserted']}")
    print(f"[evaluate_outcomes] skipped:   {result['skipped']}  (horizon not yet complete)")

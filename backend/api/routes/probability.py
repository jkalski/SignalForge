from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import ProbabilityHistory

router = APIRouter(prefix="/probability", tags=["probability"])


@router.get("/summary")
def probability_summary(
    timeframe:  str = Query("1h"),
    setup_type: Optional[str] = Query(None, description="Filter to one setup type"),
    limit:      int = Query(20, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """
    Return the latest win-rate record per setup_type for the given timeframe.

    Records are produced by the probability aggregator and reflect historical
    SignalOutcome data.  Returns an empty list until the aggregator has run
    at least once with enough samples.
    """
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    try:
        # Subquery: latest computed_at per (setup_type, timeframe).
        subq = (
            db.query(
                ProbabilityHistory.setup_type,
                func.max(ProbabilityHistory.computed_at).label("latest"),
            )
            .filter(ProbabilityHistory.timeframe == timeframe)
            .group_by(ProbabilityHistory.setup_type)
            .subquery()
        )

        q = (
            db.query(ProbabilityHistory)
            .join(
                subq,
                (ProbabilityHistory.setup_type  == subq.c.setup_type) &
                (ProbabilityHistory.computed_at == subq.c.latest),
            )
            .filter(ProbabilityHistory.timeframe == timeframe)
        )

        if setup_type:
            q = q.filter(ProbabilityHistory.setup_type == setup_type)

        rows = q.order_by(ProbabilityHistory.win_rate.desc()).limit(limit).all()

        return [
            {
                "setup_type":   r.setup_type,
                "timeframe":    r.timeframe,
                "sample_count": r.sample_count,
                "win_count":    r.win_count,
                "win_rate":     float(r.win_rate)    if r.win_rate    is not None else None,
                "avg_r_won":    float(r.avg_r_won)   if r.avg_r_won   is not None else None,
                "avg_r_lost":   float(r.avg_r_lost)  if r.avg_r_lost  is not None else None,
                "expected_r":   float(r.expected_r)  if r.expected_r  is not None else None,
                "window_start": r.window_start,
                "window_end":   r.window_end,
                "computed_at":  r.computed_at,
            }
            for r in rows
        ]

    finally:
        db.close()

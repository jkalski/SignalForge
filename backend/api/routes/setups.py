from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from backend.db.models import Setup

router = APIRouter(prefix="/setups", tags=["setups"])


@router.get("/latest")
def get_latest_setups(
    timeframe: str = Query("1h"),
    source: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    try:
        q = db.query(Setup).filter(Setup.timeframe == timeframe)
        if source:
            q = q.filter(Setup.source == source)

        rows = q.order_by(Setup.ts.desc(), Setup.score.asc()).limit(limit).all()

        return [
            {
                "id":            r.id,
                "run_id":        r.run_id,
                "symbol":        r.symbol,
                "timeframe":     r.timeframe,
                "source":        r.source,
                "ts":            r.ts,
                "close":         float(r.close)        if r.close        is not None else None,
                "ema_20":        float(r.ema_20)       if r.ema_20       is not None else None,
                "ema_50":        float(r.ema_50)       if r.ema_50       is not None else None,
                "rsi_14":        float(r.rsi_14)       if r.rsi_14       is not None else None,
                "atr_14":        float(r.atr_14)       if r.atr_14       is not None else None,
                "score":         float(r.score)        if r.score        is not None else None,
                "distance_pct":  float(r.distance_pct) if r.distance_pct is not None else None,
                "current_state": r.current_state,
                "trigger_type":  r.trigger_type,
                "created_at":    r.created_at,
            }
            for r in rows
        ]

    finally:
        db.close()

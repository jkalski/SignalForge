import statistics
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from backend.db.models import SignalOutcome

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/summary")
def performance_summary(
    timeframe: str = Query("1h"),
    horizon: int   = Query(24, description="Horizon in bars, e.g. 6, 24, 72"),
    symbol: Optional[str] = Query(None, description="Filter to one symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name (joined via signal)"),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    try:
        q = (
            db.query(SignalOutcome)
            .filter(
                SignalOutcome.timeframe    == timeframe,
                SignalOutcome.horizon_bars == horizon,
                SignalOutcome.return_pct   != None,
            )
        )
        if symbol:
            q = q.filter(SignalOutcome.symbol == symbol.upper())

        rows = q.all()

        if not rows:
            return {
                "timeframe": timeframe,
                "horizon":   horizon,
                "symbol":    symbol,
                "count":     0,
                "win_rate":  None,
                "avg_return":    None,
                "median_return": None,
            }

        returns = [float(r.return_pct) for r in rows]
        wins    = [r for r in returns if r > 0]

        return {
            "timeframe":     timeframe,
            "horizon":       horizon,
            "symbol":        symbol,
            "count":         len(returns),
            "win_rate":      round(len(wins) / len(returns), 4),
            "avg_return":    round(sum(returns) / len(returns), 4),
            "median_return": round(statistics.median(returns), 4),
        }

    finally:
        db.close()

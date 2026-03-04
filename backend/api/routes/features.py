from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from backend.db.session import get_db
from backend.db.models import Candle
from backend.features.indicators import compute_features

router = APIRouter(prefix="/features", tags=["features"])


@router.get("")
def get_features(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    source: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    lookback: int = Query(200, ge=10, le=5000, description="Recent bars to use when start/end not provided"),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()
    try:
        q = db.query(Candle).filter(
            Candle.symbol == symbol.upper(),
            Candle.timeframe == timeframe,
        )
        if source:
            q = q.filter(Candle.source == source)
        if start:
            q = q.filter(Candle.ts >= start)
        if end:
            q = q.filter(Candle.ts <= end)

        if start or end:
            items = q.order_by(Candle.ts.asc()).all()
        else:
            items = q.order_by(Candle.ts.desc()).limit(lookback).all()
            items = sorted(items, key=lambda x: x.ts)

        if not items:
            raise HTTPException(status_code=404, detail="No candles found")

        candles = [
            {
                "ts": x.ts,
                "open": float(x.open),
                "high": float(x.high),
                "low": float(x.low),
                "close": float(x.close),
                "volume": x.volume,
            }
            for x in items
        ]
    finally:
        db.close()

    return compute_features(candles)

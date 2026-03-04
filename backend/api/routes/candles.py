from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.models import Candle

router = APIRouter(prefix="/candles", tags=["candles"])

class CandleIn(BaseModel):
    symbol: str
    timeframe: str
    source: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

@router.post("")
def upsert_candles(payload: List[CandleIn], db: Session = Depends(get_db)):
    rows = 0
    for c in payload:
        obj = Candle(
            symbol=c.symbol.upper(),
            timeframe=c.timeframe,
            source=c.source,
            ts=c.ts,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
        )
        db.add(obj)
        rows += 1

    # for now: naive insert
    # next step: handle unique constraint conflicts (SQLite vs Postgres differs)
    db.commit()
    return {"inserted": rows}

@router.get("")
def list_candles(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    source: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
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

    items = q.order_by(Candle.ts.desc()).limit(limit).all()
    return [
        {
            "symbol": x.symbol,
            "timeframe": x.timeframe,
            "source": x.source,
            "ts": x.ts,
            "open": float(x.open),
            "high": float(x.high),
            "low": float(x.low),
            "close": float(x.close),
            "volume": x.volume,
        }
        for x in items
    ]
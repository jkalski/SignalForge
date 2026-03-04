import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.models import Candle, Signal
from backend.features.indicators import compute_features

router = APIRouter(prefix="/signals", tags=["signals"])

ATR_STOP_MULT   = 1.5
ATR_TARGET_MULT = 3.0
R_MULTIPLE      = round(ATR_TARGET_MULT / ATR_STOP_MULT, 2)  # 2.0


@router.get("/simple")
def get_simple_signal(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    source: Optional[str] = Query(None),
    lookback: int = Query(200, ge=52, le=5000, description="Bars of history; min 52 to warm up EMA50"),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()
    sym = symbol.upper()

    try:
        q = db.query(Candle).filter(
            Candle.symbol == sym,
            Candle.timeframe == timeframe,
        )
        if source:
            q = q.filter(Candle.source == source)

        items = q.order_by(Candle.ts.desc()).limit(lookback).all()
        items = sorted(items, key=lambda x: x.ts)

        if len(items) < 52:
            raise HTTPException(
                status_code=422,
                detail=f"Need at least 52 candles to compute EMA50, got {len(items)}",
            )

        candles = [
            {
                "ts":     x.ts,
                "open":   float(x.open),
                "high":   float(x.high),
                "low":    float(x.low),
                "close":  float(x.close),
                "volume": x.volume,
            }
            for x in items
        ]

        rows = compute_features(candles)
        curr = rows[-1]
        prev = rows[-2]

        ema20_c = curr["ema_20"]
        ema50_c = curr["ema_50"]
        ema20_p = prev["ema_20"]
        ema50_p = prev["ema_50"]
        rsi     = curr["rsi_14"]
        atr     = curr["atr_14"] or 0
        close   = curr["close"]
        ts: datetime = curr["ts"]

        golden = (
            ema20_p is not None and ema50_p is not None
            and ema20_p <= ema50_p and ema20_c > ema50_c
        )
        death = (
            ema20_p is not None and ema50_p is not None
            and ema20_p >= ema50_p and ema20_c < ema50_c
        )

        if golden:
            direction    = "BUY"
            db_direction = "long"
            trigger      = "golden_cross"
            entry        = close
            stop         = round(close - ATR_STOP_MULT * atr, 4)
            target       = round(close + ATR_TARGET_MULT * atr, 4)
        elif death:
            direction    = "SELL"
            db_direction = "short"
            trigger      = "death_cross"
            entry        = close
            stop         = round(close + ATR_STOP_MULT * atr, 4)
            target       = round(close - ATR_TARGET_MULT * atr, 4)
        else:
            direction    = "NONE"
            db_direction = None
            trigger      = None
            entry        = close
            stop         = None
            target       = None

        signal_id = None

        if db_direction is not None:
            # Deterministic ID: one signal per symbol/timeframe/bar
            ts_tag    = ts.strftime("%Y%m%dT%H%M%S") if isinstance(ts, datetime) else str(ts)
            signal_id = f"{sym}_{timeframe}_{ts_tag}_{trigger}"

            existing = db.get(Signal, signal_id)
            if existing is None:
                db.add(Signal(
                    id           = signal_id,
                    symbol       = sym,
                    timeframe    = timeframe,
                    direction    = db_direction,
                    setup_type   = trigger,
                    entry_price  = entry,
                    stop_price   = stop,
                    target_price = target,
                    r_multiple   = R_MULTIPLE,
                    status       = "active",
                    context_snapshot = json.dumps({
                        "ema_20": ema20_c,
                        "ema_50": ema50_c,
                        "rsi_14": rsi,
                        "atr_14": atr,
                        "source": source,
                    }),
                    created_at   = datetime.utcnow(),
                ))
                db.commit()

        return {
            "symbol":     sym,
            "timeframe":  timeframe,
            "signal":     direction,
            "trigger":    trigger,
            "signal_id":  signal_id,
            "ts":         ts,
            "close":      close,
            "ema_20":     ema20_c,
            "ema_50":     ema50_c,
            "rsi_14":     rsi,
            "atr_14":     atr,
            "entry":      entry,
            "stop":       stop,
            "target":     target,
            "r_multiple": R_MULTIPLE,
        }

    finally:
        db.close()

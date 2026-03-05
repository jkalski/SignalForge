import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.models import Candle, Signal
from backend.agent.pipeline import run_structure_pipeline, R_MULTIPLE

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/simple")
def get_simple_signal(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    source: Optional[str] = Query(None),
    lookback: int = Query(200, ge=52, le=5000,
                          description="Bars of history; min 52 for indicator warmup"),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()
    sym = symbol.upper()

    try:
        q = db.query(Candle).filter(
            Candle.symbol    == sym,
            Candle.timeframe == timeframe,
        )
        if source:
            q = q.filter(Candle.source == source)

        items = q.order_by(Candle.ts.desc()).limit(lookback).all()
        items = sorted(items, key=lambda x: x.ts)

        if len(items) < 52:
            raise HTTPException(
                status_code=422,
                detail=f"Need at least 52 candles for indicator warmup, got {len(items)}",
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

        result = run_structure_pipeline(candles, symbol=sym,
                                        timeframe=timeframe, source=source)

        signal_id = None

        if result is not None and result["signal_valid"]:
            ts     = result["ts"]
            ts_tag = ts.strftime("%Y%m%dT%H%M%S") if isinstance(ts, datetime) else str(ts)
            signal_id = f"{sym}_{timeframe}_{ts_tag}_{result['event_type']}"

            if db.get(Signal, signal_id) is None:
                db.add(Signal(
                    id               = signal_id,
                    symbol           = sym,
                    timeframe        = timeframe,
                    direction        = result["direction"],
                    setup_type       = result["event_type"],
                    strategy         = "structure_v1",
                    entry_price      = result["close"],
                    stop_price       = result["stop_price"],
                    target_price     = result["target_price"],
                    r_multiple       = R_MULTIPLE,
                    status           = "active",
                    context_snapshot = json.dumps({
                        "event_type":   result["event_type"],
                        "zone_center":  result["zone_center"],
                        "zone_touches": result["zone_touches"],
                        "vol_spike":    result["vol_spike"],
                        "vol_ratio":    result["vol_ratio"],
                        "trend":        result["trend"],
                        "ema_confirms": result["ema_confirms"],
                        "ema_20":       result["ema_20"],
                        "ema_50":       result["ema_50"],
                        "rsi_14":       result["rsi_14"],
                        "atr_14":       result["atr_14"],
                        "confidence":   result["confidence"],
                        "source":       source,
                    }),
                    created_at = datetime.utcnow(),
                ))
                db.commit()

        # Response shape preserved from previous version; new fields appended.
        if result is not None:
            direction_label = {"long": "BUY", "short": "SELL"}.get(
                result["direction"], "NONE"
            )
            return {
                # Legacy fields (same keys, structure-pipeline values)
                "symbol":     sym,
                "timeframe":  timeframe,
                "signal":     direction_label if result["signal_valid"] else "NONE",
                "trigger":    result["event_type"],
                "signal_id":  signal_id,
                "ts":         result["ts"],
                "close":      result["close"],
                "ema_20":     result["ema_20"],
                "ema_50":     result["ema_50"],
                "rsi_14":     result["rsi_14"],
                "atr_14":     result["atr_14"],
                "entry":      result["close"],
                "stop":       result["stop_price"],
                "target":     result["target_price"],
                "r_multiple": R_MULTIPLE,
                "strategy":   "structure_v1",
                # Structure-specific fields
                "event_type":   result["event_type"],
                "zone_center":  result["zone_center"],
                "zone_touches": result["zone_touches"],
                "vol_spike":    result["vol_spike"],
                "vol_ratio":    result["vol_ratio"],
                "trend":        result["trend"],
                "ema_confirms": result["ema_confirms"],
                "confidence":   result["confidence"],
                "signal_valid": result["signal_valid"],
            }

        # No structural event detected on this symbol's last bar.
        return {
            "symbol":       sym,
            "timeframe":    timeframe,
            "signal":       "NONE",
            "trigger":      None,
            "signal_id":    None,
            "ts":           None,
            "close":        None,
            "ema_20":       None,
            "ema_50":       None,
            "rsi_14":       None,
            "atr_14":       None,
            "entry":        None,
            "stop":         None,
            "target":       None,
            "r_multiple":   None,
            "strategy":     "structure_v1",
            "event_type":   None,
            "zone_center":  None,
            "zone_touches": None,
            "vol_spike":    False,
            "vol_ratio":    None,
            "trend":        None,
            "ema_confirms": False,
            "confidence":   0.0,
            "signal_valid": False,
        }

    finally:
        db.close()

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from backend.db.models import Candle
from backend.marketdata.universe import SYMBOLS
from backend.agent.pipeline import run_structure_pipeline

router = APIRouter(prefix="/scan", tags=["scan"])


def _score_symbol(
    db: Session,
    symbol: str,
    timeframe: str,
    source: Optional[str],
    lookback: int,
) -> Optional[Dict[str, Any]]:
    """Run the structure pipeline for one symbol.

    Returns a scored dict or None when data is insufficient or no structural
    event is detected on the last bar.

    The returned dict is shaped for the /scan/simple response and for the agent
    runner.  Legacy field names (score, score_raw, current_state, trigger_type,
    triggered) are preserved so existing consumers are not broken; their values
    now carry structure-pipeline data rather than EMA-gap data.
    """
    q = db.query(Candle).filter(
        Candle.symbol    == symbol,
        Candle.timeframe == timeframe,
    )
    if source:
        q = q.filter(Candle.source == source)

    items = q.order_by(Candle.ts.desc()).limit(lookback).all()
    if len(items) < 52:
        return None

    items = sorted(items, key=lambda x: x.ts)
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

    result = run_structure_pipeline(candles, symbol=symbol,
                                    timeframe=timeframe, source=source)
    if result is None:
        return None

    # Expose pipeline fields directly and alias legacy keys so that any
    # existing dashboard / API client keeps receiving familiar field names.
    return {
        **result,
        # Legacy aliases (backwards-compatible)
        "score_raw":     result["zone_touches"],
        "score":         result["confidence"],
        "current_state": result["trend"],
        "trigger_type":  result["event_type"],
        "triggered":     result["signal_valid"],
    }


@router.get("/simple")
def scan_simple(
    timeframe: str = Query("5m"),
    source: Optional[str] = Query(None),
    lookback: int = Query(200, ge=52, le=5000),
    top_n: int = Query(10, ge=1, le=50,
                       description="Max candidates returned (ranked by confidence)"),
    symbols: Optional[str] = Query(
        None, description="Comma-separated override; defaults to major universe"
    ),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    universe = (
        [s.strip().upper() for s in symbols.split(",")]
        if symbols
        else SYMBOLS
    )

    scored: List[Dict[str, Any]] = []
    try:
        for sym in universe:
            row = _score_symbol(db, sym, timeframe, source, lookback)
            if row:
                scored.append(row)
    finally:
        db.close()

    # Rank by confidence descending; tiebreak on zone_touches then symbol.
    scored.sort(key=lambda x: (x["confidence"], x["zone_touches"], x["symbol"]),
                reverse=True)

    candidates = scored[:top_n]
    # Triggers = candidates where all gates pass (vol_spike + ema_confirms).
    triggers = [r for r in scored if r["triggered"]]

    return {
        "timeframe":  timeframe,
        "source":     source,
        "scanned":    len(universe),
        "candidates": candidates,
        "triggers":   triggers,
    }

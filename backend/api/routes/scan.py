from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from backend.db.models import Candle
from backend.marketdata.universe import SYMBOLS
from backend.agent.pipeline import run_structure_pipeline

router = APIRouter(prefix="/scan", tags=["scan"])

# LTF → HTF timeframe mapping (mirrors backend.agent.runner._HTF_MAP).
_HTF_MAP: Dict[str, str] = {
    "1m":  "15m",
    "5m":  "1h",
    "15m": "4h",
    "1h":  "4h",
    "4h":  "1d",
    "1d":  "1w",
}
_HTF_LOOKBACK = 200


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

    # Fetch HTF candles for MTF enrichment (same source, coarser timeframe).
    htf_tf      = _HTF_MAP.get(timeframe)
    htf_candles: List[Dict[str, Any]] = []
    if htf_tf:
        htf_q = db.query(Candle).filter(
            Candle.symbol    == symbol,
            Candle.timeframe == htf_tf,
        )
        if source:
            htf_q = htf_q.filter(Candle.source == source)
        htf_items = htf_q.order_by(Candle.ts.desc()).limit(_HTF_LOOKBACK).all()
        htf_items = sorted(htf_items, key=lambda x: x.ts)
        htf_candles = [
            {
                "ts":     x.ts,
                "open":   float(x.open),
                "high":   float(x.high),
                "low":    float(x.low),
                "close":  float(x.close),
                "volume": x.volume,
            }
            for x in htf_items
        ]

    result = run_structure_pipeline(
        candles,
        symbol      = symbol,
        timeframe   = timeframe,
        source      = source,
        htf_candles = htf_candles or None,
    )
    if result is None:
        return None

    # Enrich data_quality with HTF availability metadata.
    dq = result.get("data_quality") or {}
    dq["htf_timeframe"]  = htf_tf
    dq["htf_bars_count"] = len(htf_candles)
    dq["htf_last_ts"]    = htf_candles[-1]["ts"] if htf_candles else None
    dq["htf_provider"]   = source or "unknown"
    result["data_quality"] = dq

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

    # Rank by confluence_score descending; tiebreak on confidence then symbol.
    scored.sort(
        key=lambda x: (x.get("confluence_score", 0), x["confidence"], x["symbol"]),
        reverse=True,
    )

    candidates = scored[:top_n]
    # Triggers = all symbols (not just top_n) where signal_status is "active".
    triggers = [r for r in scored if r.get("signal_status") == "active"]

    return {
        "timeframe":  timeframe,
        "source":     source,
        "scanned":    len(universe),
        "candidates": candidates,
        "triggers":   triggers,
    }

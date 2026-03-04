import math

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

from backend.db.models import Candle
from backend.features.indicators import compute_features

router = APIRouter(prefix="/scan", tags=["scan"])

MAJOR_SYMBOLS: List[str] = [
    # Broad ETFs
    "SPY", "QQQ", "DIA", "IWM",
    # Sector ETFs
    "XLK", "XLF", "XLE",
    # Semis
    "SMH", "SOXX",
    # Macro / fixed income
    "GLD", "TLT", "HYG",
    # Mega-cap stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX",
]


def _score_symbol(
    db: Session,
    symbol: str,
    timeframe: str,
    source: Optional[str],
    lookback: int,
) -> Optional[Dict[str, Any]]:
    """Return a scored dict for one symbol, or None if data is insufficient.

    score = abs(ema_20 - ema_50) / atr_14
    Lower score means the two EMAs are closer together (nearer to crossover).
    """
    q = db.query(Candle).filter(
        Candle.symbol == symbol,
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

    rows = compute_features(candles)
    curr = rows[-1]
    prev = rows[-2]

    ema20_c = curr["ema_20"]
    ema50_c = curr["ema_50"]
    ema20_p = prev["ema_20"]
    ema50_p = prev["ema_50"]
    atr     = curr["atr_14"]
    close   = curr["close"]

    def _ok(v) -> bool:
        return v is not None and math.isfinite(v)

    if not (_ok(ema20_c) and _ok(ema50_c) and _ok(atr) and atr > 0
            and _ok(close) and close > 0):
        return None

    gap           = abs(ema20_c - ema50_c)
    score_raw     = round(gap, 4)
    score         = round(gap / atr, 4)
    distance_pct  = round(gap / close * 100, 4)
    current_state = "bullish" if ema20_c > ema50_c else "bearish"

    if ema20_p is not None and ema50_p is not None:
        if ema20_p <= ema50_p and ema20_c > ema50_c:
            trigger_type = "golden"
        elif ema20_p >= ema50_p and ema20_c < ema50_c:
            trigger_type = "death"
        else:
            trigger_type = None
    else:
        trigger_type = None

    return {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "source":        source,
        "ts":            curr["ts"],
        "close":         close,
        "ema_20":        ema20_c,
        "ema_50":        ema50_c,
        "rsi_14":        curr["rsi_14"],
        "atr_14":        atr,
        "score_raw":     score_raw,
        "score":         score,
        "distance_pct":  distance_pct,
        "current_state": current_state,
        "trigger_type":  trigger_type,
        "triggered":     trigger_type is not None,
    }


@router.get("/simple")
def scan_simple(
    timeframe: str = Query("5m"),
    source: Optional[str] = Query(None),
    lookback: int = Query(200, ge=52, le=5000),
    top_n: int = Query(10, ge=1, le=50, description="Max candidates returned (ranked by score)"),
    symbols: Optional[str] = Query(None, description="Comma-separated override; defaults to major universe"),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    universe = (
        [s.strip().upper() for s in symbols.split(",")]
        if symbols
        else MAJOR_SYMBOLS
    )

    scored: List[Dict[str, Any]] = []
    try:
        for sym in universe:
            row = _score_symbol(db, sym, timeframe, source, lookback)
            if row:
                scored.append(row)
    finally:
        db.close()

    # Primary: score asc (lower = EMAs tighter = nearer to crossover)
    # Secondary: distance_pct asc (price-normalised tiebreaker)
    # Tertiary: symbol asc (deterministic output between calls)
    scored.sort(key=lambda x: (x["score"], x["distance_pct"], x["symbol"]))

    candidates = scored[:top_n]
    # Triggers come from all scored symbols (not just top_n) so a recently-
    # fired cross that has since diverged isn't accidentally suppressed.
    triggers = [r for r in scored if r["triggered"]]

    return {
        "timeframe":  timeframe,
        "source":     source,
        "scanned":    len(universe),
        "candidates": candidates,
        "triggers":   triggers,
    }

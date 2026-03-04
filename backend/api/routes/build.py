from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import pandas as pd

from backend.db.session import get_db
from backend.db.models import Candle

router = APIRouter(prefix="/build", tags=["build"])

# pandas 3.x frequency aliases
TF_TO_FREQ = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

# which aggregations are valid (must be strictly larger timeframe)
VALID_UPSAMPLES = {
    "1m":  {"5m", "15m", "1h", "4h", "1d"},
    "5m":  {"15m", "1h", "4h", "1d"},
    "15m": {"1h", "4h", "1d"},
    "1h":  {"4h", "1d"},
}


@router.post("/aggregate")
def aggregate_candles(
    symbol: str = Query(...),
    from_tf: str = Query(..., description="Source timeframe, e.g. 5m"),
    to_tf: str = Query(..., description="Target timeframe, e.g. 15m or 1h"),
    source: str = Query("alpaca", description="Source label on the existing candles"),
    db: Session = Depends(get_db),
):
    if from_tf not in TF_TO_FREQ:
        raise HTTPException(400, detail=f"Unknown from_tf. Valid: {list(TF_TO_FREQ)}")
    if to_tf not in TF_TO_FREQ:
        raise HTTPException(400, detail=f"Unknown to_tf. Valid: {list(TF_TO_FREQ)}")
    if to_tf not in VALID_UPSAMPLES.get(from_tf, set()):
        raise HTTPException(400, detail=f"Cannot aggregate {from_tf} → {to_tf}")

    sym = symbol.upper()
    agg_source = f"agg_{source}"  # e.g. agg_alpaca

    rows = (
        db.query(Candle)
        .filter(
            Candle.symbol == sym,
            Candle.timeframe == from_tf,
            Candle.source == source,
        )
        .order_by(Candle.ts.asc())
        .all()
    )

    if not rows:
        return {"aggregated": 0, "skipped": 0, "detail": f"No {from_tf}/{source} candles for {sym}"}

    df = pd.DataFrame([
        {
            "ts":     r.ts,
            "open":   float(r.open),
            "high":   float(r.high),
            "low":    float(r.low),
            "close":  float(r.close),
            "volume": int(r.volume) if r.volume else 0,
        }
        for r in rows
    ])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    # Resample: left-closed, left-labeled = standard bar convention
    agg = (
        df.resample(TF_TO_FREQ[to_tf], closed="left", label="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )

    if agg.empty:
        return {"aggregated": 0, "skipped": 0}

    ts_list = [t.to_pydatetime() for t in agg["ts"]]

    existing_ts = set(
        t[0]
        for t in db.query(Candle.ts)
        .filter(
            Candle.symbol == sym,
            Candle.timeframe == to_tf,
            Candle.source == agg_source,
            Candle.ts.in_(ts_list),
        )
        .all()
    )

    inserted = 0
    skipped = 0

    for _, r in agg.iterrows():
        ts = r["ts"].to_pydatetime()
        if ts in existing_ts:
            skipped += 1
            continue
        db.add(Candle(
            symbol=sym,
            timeframe=to_tf,
            source=agg_source,
            ts=ts,
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=int(r["volume"]) if r["volume"] > 0 else None,
        ))
        inserted += 1

    db.commit()
    return {
        "symbol":     sym,
        "from_tf":    from_tf,
        "to_tf":      to_tf,
        "source":     agg_source,
        "aggregated": inserted,
        "skipped":    skipped,
    }

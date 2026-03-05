from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import requests
from backend.db.models import IngestRun
import csv
from io import StringIO

from backend.db.session import get_db
from backend.db.models import Candle
from backend.config import settings
from backend.marketdata.yahoo import fetch_yahoo_bars, YAHOO_INTERVAL_MAP
from backend.marketdata.universe import SYMBOLS

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALPACA_DATA_URL = "https://data.alpaca.markets"

ALPACA_INTERVAL_MAP = {
    "1m":  "1Min",
    "5m":  "5Min",
    "15m": "15Min",
    "1h":  "1Hour",
    "1d":  "1Day",
}

STOOQ_INTERVAL_MAP = {
    "1d": ("d", "1d"),
    "1w": ("w", "1w"),
    "1m": ("m", "1m"),
}

def _stooq_symbol(symbol: str) -> str:
    # Simple US mapping: SPY -> spy.us
    # Later: detect if user already passed ".us"
    s = symbol.strip().lower()
    if "." not in s:
        s = f"{s}.us"
    return s

@router.post("/stooq")
def ingest_stooq(
    symbol: str = Query(..., description="Ticker, e.g. SPY"),
    timeframe: str = Query("1d", description="One of 1d, 1w, 1m (Stooq only)"),
    db: Session = Depends(get_db),
):
    started_at = datetime.utcnow()
    stored_timeframe = "1d"

    if timeframe not in STOOQ_INTERVAL_MAP:
        raise HTTPException(status_code=400, detail="Stooq supports only timeframe=1d,1w,1m")

    interval_param, stored_timeframe = STOOQ_INTERVAL_MAP[timeframe]
    stooq_sym = _stooq_symbol(symbol)

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i={interval_param}"
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200 or not resp.text:
        raise HTTPException(status_code=400, detail="Failed to fetch from Stooq")

    reader = csv.DictReader(StringIO(resp.text))
    ts_list = []

    parsed_rows = []
    for row in reader:
        # Stooq CSV columns: Date, Open, High, Low, Close, Volume
        if not row.get("Date"):
            continue
        if row.get("Close") in (None, "", "0"):
            continue

        ts = datetime.strptime(row["Date"], "%Y-%m-%d")
        ts_list.append(ts)

        parsed_rows.append(
            {
                "ts": ts,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if row.get("Volume") else None,
            }
        )

    if not parsed_rows:
        return {"inserted": 0, "skipped": 0}

    # Load existing timestamps for this (symbol, timeframe, source)
    existing_ts = set(
        t[0]
        for t in db.query(Candle.ts)
        .filter(
            Candle.symbol == symbol.upper(),
            Candle.timeframe == stored_timeframe,
            Candle.source == "stooq",
            Candle.ts.in_(ts_list),
        )
        .all()
    )

    inserted = 0
    skipped = 0

    for r in parsed_rows:
        if r["ts"] in existing_ts:
            skipped += 1
            continue

        db.add(
            Candle(
                symbol=symbol.upper(),
                timeframe=stored_timeframe,
                source="stooq",
                ts=r["ts"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
            )
        )
        inserted += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        # log failed ingest run
        db.add(
            IngestRun(
                run_type=f"stooq_{stored_timeframe}",
                status="error",
                symbols=symbol.upper(),
                rows_written=inserted,
                error_msg=str(e),
                started_at=started_at,
                finished_at=datetime.utcnow(),
            )
        )
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    # log successful ingest run
    finished_at = datetime.utcnow()
    db.add(
        IngestRun(
            run_type=f"stooq_{stored_timeframe}",
            status="ok",
            symbols=symbol.upper(),
            rows_written=inserted,
            started_at=started_at,
            finished_at=finished_at,
        )
    )
    db.commit()

    return {"inserted": inserted, "skipped": skipped}


@router.post("/alpaca")
def ingest_alpaca(
    symbol: str = Query(..., description="Ticker, e.g. AAPL"),
    timeframe: str = Query("1h", description="One of 1m, 5m, 15m, 1h, 1d"),
    start: Optional[str] = Query(None, description="ISO8601 start, e.g. 2024-01-01"),
    end: Optional[str] = Query(None, description="ISO8601 end, e.g. 2024-12-31"),
    max_bars: Optional[int] = Query(None, ge=1, le=100000, description="Cap total bars fetched across all pages"),
    db: Session = Depends(get_db),
):
    if timeframe not in ALPACA_INTERVAL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Alpaca supports timeframes: {list(ALPACA_INTERVAL_MAP)}",
        )

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise HTTPException(status_code=500, detail="Alpaca credentials not configured")

    sym_upper = symbol.upper()
    alpaca_tf = ALPACA_INTERVAL_MAP[timeframe]
    started_at = datetime.utcnow()

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    params: dict = {"timeframe": alpaca_tf, "limit": 10000}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    # Paginate through all results
    all_bars = []
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{sym_upper}/bars",
            headers=headers,
            params=params,
            timeout=30,
        )

        if resp.status_code == 403:
            raise HTTPException(status_code=403, detail="Alpaca: unauthorized or subscription required")
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Alpaca API error: {resp.text[:300]}")

        data = resp.json()
        all_bars.extend(data.get("bars") or [])
        if max_bars and len(all_bars) >= max_bars:
            all_bars = all_bars[:max_bars]
            break
        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not all_bars:
        return {"inserted": 0, "skipped": 0}

    # Normalize Alpaca fields: t, o, h, l, c, v
    parsed_rows = []
    ts_list = []

    for bar in all_bars:
        ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).replace(tzinfo=None)
        ts_list.append(ts)
        parsed_rows.append({
            "ts": ts,
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": int(bar["v"]) if bar.get("v") else None,
        })

    # Dedup against existing rows
    existing_ts = set(
        t[0]
        for t in db.query(Candle.ts)
        .filter(
            Candle.symbol == sym_upper,
            Candle.timeframe == timeframe,
            Candle.source == "alpaca",
            Candle.ts.in_(ts_list),
        )
        .all()
    )

    inserted = 0
    skipped = 0

    for r in parsed_rows:
        if r["ts"] in existing_ts:
            skipped += 1
            continue
        db.add(Candle(
            symbol=sym_upper,
            timeframe=timeframe,
            source="alpaca",
            ts=r["ts"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
        ))
        inserted += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        db.add(IngestRun(
            run_type=f"alpaca_{timeframe}",
            status="error",
            symbols=sym_upper,
            rows_written=inserted,
            error_msg=str(e),
            started_at=started_at,
            finished_at=datetime.utcnow(),
        ))
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    db.add(IngestRun(
        run_type=f"alpaca_{timeframe}",
        status="ok",
        symbols=sym_upper,
        rows_written=inserted,
        started_at=started_at,
        finished_at=datetime.utcnow(),
    ))
    db.commit()

    return {"inserted": inserted, "skipped": skipped}


def _ingest_yahoo_one(
    db: Session,
    sym_upper: str,
    timeframe: str,
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    max_bars: Optional[int],
) -> dict:
    """Fetch and store Yahoo bars for one symbol. Returns result dict; raises on error."""
    import math

    def _finite(v) -> bool:
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    bars = fetch_yahoo_bars(
        symbol=sym_upper,
        timeframe=timeframe,
        period=period,
        start=start,
        end=end,
        max_bars=max_bars,
    )

    # Drop bars with any non-finite OHLC (Yahoo sometimes emits NaN on bad/partial bars)
    bars = [
        b for b in bars
        if _finite(b["open"]) and _finite(b["high"]) and _finite(b["low"]) and _finite(b["close"])
    ]

    if not bars:
        suggested = {
            "1m": "5d", "5m": "60d", "15m": "60d", "1h": "6mo", "1d": "2y"
        }.get(timeframe, "1mo")
        return {
            "inserted": 0,
            "skipped": 0,
            "warning": (
                f"Yahoo returned no bars for {sym_upper} {timeframe}. "
                f"Try a smaller period (e.g. period={suggested}). "
                f"Intraday limits: 1m≤5d, 5m/15m≤60d, 1h≤730d."
            ),
        }

    ts_list = [b["ts"] for b in bars]

    existing_ts = set(
        t[0]
        for t in db.query(Candle.ts)
        .filter(
            Candle.symbol == sym_upper,
            Candle.timeframe == timeframe,
            Candle.source == "yahoo",
            Candle.ts.in_(ts_list),
        )
        .all()
    )

    inserted = 0
    skipped = 0

    for b in bars:
        if b["ts"] in existing_ts:
            skipped += 1
            continue
        db.add(Candle(
            symbol=sym_upper,
            timeframe=timeframe,
            source="yahoo",
            ts=b["ts"],
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            volume=b["volume"],
        ))
        inserted += 1

    db.commit()
    return {"inserted": inserted, "skipped": skipped}


@router.post("/yahoo")
def ingest_yahoo(
    symbol: str = Query(..., description="Ticker, e.g. AAPL"),
    timeframe: str = Query("1h", description="One of 1m, 5m, 15m, 1h, 1d"),
    period: Optional[str] = Query(None, description="yfinance period: 5d,1mo,3mo,6mo,1y,2y (default based on timeframe)"),
    start: Optional[str] = Query(None, description="ISO8601 start date, e.g. 2024-01-01"),
    end: Optional[str] = Query(None, description="ISO8601 end date, e.g. 2024-12-31"),
    max_bars: Optional[int] = Query(None, ge=1, le=100000, description="Cap total bars fetched"),
    db: Session = Depends(get_db),
):
    if timeframe not in YAHOO_INTERVAL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Yahoo supports timeframes: {list(YAHOO_INTERVAL_MAP)}",
        )

    sym_upper = symbol.upper()
    started_at = datetime.utcnow()

    try:
        result = _ingest_yahoo_one(db, sym_upper, timeframe, period, start, end, max_bars)
    except Exception as e:
        db.rollback()
        db.add(IngestRun(
            run_type=f"yahoo_{timeframe}",
            status="error",
            symbols=sym_upper,
            rows_written=0,
            error_msg=str(e),
            started_at=started_at,
            finished_at=datetime.utcnow(),
        ))
        db.commit()
        raise HTTPException(status_code=500, detail=f"Yahoo fetch error: {e}")

    db.add(IngestRun(
        run_type=f"yahoo_{timeframe}",
        status="ok",
        symbols=sym_upper,
        rows_written=result["inserted"],
        started_at=started_at,
        finished_at=datetime.utcnow(),
    ))
    db.commit()

    return result


@router.post("/yahoo/universe")
def ingest_yahoo_universe(
    timeframe: str = Query("1h", description="One of 1m, 5m, 15m, 1h, 1d"),
    period: Optional[str] = Query(None, description="yfinance period: 5d,1mo,3mo,6mo,1y,2y"),
    start: Optional[str] = Query(None, description="ISO8601 start date, e.g. 2024-01-01"),
    end: Optional[str] = Query(None, description="ISO8601 end date, e.g. 2024-12-31"),
    db: Session = Depends(get_db),
):
    """Ingest Yahoo bars for every symbol in the universe list."""
    if timeframe not in YAHOO_INTERVAL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Yahoo supports timeframes: {list(YAHOO_INTERVAL_MAP)}",
        )

    started_at = datetime.utcnow()
    total_inserted = 0
    total_skipped = 0
    errors: list = []

    for symbol in SYMBOLS:
        try:
            result = _ingest_yahoo_one(db, symbol, timeframe, period, start, end, None)
            total_inserted += result["inserted"]
            total_skipped += result["skipped"]
        except Exception as e:
            db.rollback()
            errors.append({"symbol": symbol, "error": str(e)})

    db.add(IngestRun(
        run_type=f"yahoo_{timeframe}_universe",
        status="ok" if not errors else "partial",
        symbols=",".join(SYMBOLS),
        rows_written=total_inserted,
        error_msg=("; ".join(f"{e['symbol']}: {e['error']}" for e in errors) or None),
        started_at=started_at,
        finished_at=datetime.utcnow(),
    ))
    db.commit()

    return {
        "symbols": len(SYMBOLS),
        "inserted": total_inserted,
        "skipped": total_skipped,
        "errors": errors,
    }


@router.get("/runs")
def list_ingest_runs(
    status: Optional[str] = Query(None, description="Filter by status: ok or error"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    q = db.query(IngestRun)
    if status:
        q = q.filter(IngestRun.status == status)

    runs = q.order_by(IngestRun.started_at.desc()).limit(limit).all()

    def _row(r: IngestRun):
        return {
            "id": r.id,
            "run_type": r.run_type,
            "status": r.status,
            "symbols": r.symbols,
            "rows_written": r.rows_written,
            "error_msg": r.error_msg,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }

    return [_row(r) for r in runs]
"""backend/agent/runner.py

Structure-first trading agent loop: scan → zone detection → event classification
→ gate checks (volume + EMA) → generate signals → log run.

Usage:
    python -m backend.agent.runner
    python -m backend.agent.runner --timeframe 1h --source yahoo --top-n 10
    python -m backend.agent.runner --symbols AAPL,MSFT,NVDA --timeframe 15m
"""

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from datetime import UTC, datetime
from functools import partial
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.marketdata.universe import SYMBOLS
from backend.agent.pipeline import run_structure_pipeline, R_MULTIPLE
from backend.db.models import AgentRun, Candle, Setup, Signal
from backend.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Setup.trigger_type is VARCHAR(10) — use short codes for new event types.
_TRIGGER_SHORT: Dict[str, str] = {
    "breakout_up":    "brk_up",
    "breakdown_down": "brk_down",
    "bounce_up":      "bnc_up",
    "reject_down":    "rej_down",
}

# Parallel workers for the scan phase.
# Each worker owns its own DB session; SQLAlchemy's default pool handles up to
# pool_size(5) + max_overflow(10) = 15 simultaneous connections safely.
_SCAN_WORKERS = 8


def run_agent(
    timeframe: str = "1h",
    source: Optional[str] = "yahoo",
    lookback: int = 200,
    top_n: int = 10,
    symbols: Optional[List[str]] = None,
) -> dict:
    """Execute one full structure-first agent cycle.

    Pipeline per symbol (parallel):
      1. Fetch candles from DB in a dedicated session.
      2. run_structure_pipeline: pivots → zones → volume → trend → event.

    Main thread (sequential):
      3. Gate: valid only when event + vol_spike + ema_confirms.
      4. Rank all events by confidence descending, take top_n.
      5. Persist top_n as Setup rows.
      6. Create Signal rows for valid (gated) setups only.
      7. Write AgentRun audit record.

    Returns dict with:
        status, scanned, candidates_considered, signals_created,
        zones_detected, scan_time_ms
    """
    started_at = datetime.now(UTC)
    universe   = symbols or SYMBOLS

    logger.info("agent run starting | symbols=%d | timeframe=%s | source=%s",
                len(universe), timeframe, source)

    db = SessionLocal()

    try:
        # ── 0. Open run record ────────────────────────────────────────────────
        run = AgentRun(
            started_at            = started_at,
            timeframe             = timeframe,
            source                = source,
            scanned               = 0,
            candidates_considered = 0,
            signals_created       = 0,
            status                = "running",
        )
        db.add(run)
        db.flush()  # populate run.id before writing Setup rows

        # ── 1. Parallel scan across all symbols ───────────────────────────────
        worker = partial(
            _scan_one_symbol,
            timeframe = timeframe,
            source    = source,
            lookback  = lookback,
        )

        scan_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers = min(_SCAN_WORKERS, len(universe))
        ) as pool:
            raw = list(pool.map(worker, universe))
        scan_time_ms = round((time.monotonic() - scan_start) * 1000, 1)

        results: List[Dict[str, Any]] = [r for r in raw if r is not None]

        # ── 2. Aggregate scan metrics ─────────────────────────────────────────
        zones_detected = sum(r.get("zones_detected", 0) for r in results)

        logger.info(
            "scan complete | symbols=%d | with_events=%d | zones=%d | %.0fms",
            len(universe), len(results), zones_detected, scan_time_ms,
        )

        # ── 3. Rank by confidence desc, take top_n ────────────────────────────
        results.sort(key=lambda r: r["confidence"], reverse=True)
        candidates = results[:top_n]

        # ── 4. Persist top candidates as Setup rows ───────────────────────────
        for c in candidates:
            ts     = c["ts"]
            ts_tag = ts.strftime("%Y%m%dT%H%M%S") if isinstance(ts, datetime) else str(ts)
            src    = source or "none"
            setup_id = f"{run.id}_{c['symbol']}_{timeframe}_{src}_{ts_tag}"

            if db.get(Setup, setup_id) is None:
                db.add(Setup(
                    id            = setup_id,
                    run_id        = run.id,
                    symbol        = c["symbol"],
                    timeframe     = timeframe,
                    source        = source,
                    ts            = ts,
                    close         = c["close"],
                    ema_20        = c["ema_20"],
                    ema_50        = c["ema_50"],
                    rsi_14        = c["rsi_14"],
                    atr_14        = c["atr_14"],
                    score_raw     = c["zone_touches"],
                    score         = c["confidence"],
                    distance_pct  = c["distance_pct"],
                    current_state = c["trend"],
                    trigger_type  = _TRIGGER_SHORT.get(c["event_type"], c["event_type"][:10]),
                    created_at    = datetime.now(UTC),
                ))

        # ── 5. Generate Signal rows for valid (gated) setups ──────────────────
        signals_created = 0

        for c in candidates:
            if not c["signal_valid"]:
                continue

            event_type = c["event_type"]
            direction  = c["direction"]
            close      = c["close"]
            ts         = c["ts"]
            ts_tag     = ts.strftime("%Y%m%dT%H%M%S") if isinstance(ts, datetime) else str(ts)
            signal_id  = f"{c['symbol']}_{timeframe}_{ts_tag}_{event_type}"

            if db.get(Signal, signal_id) is None:
                db.add(Signal(
                    id               = signal_id,
                    symbol           = c["symbol"],
                    timeframe        = timeframe,
                    direction        = direction,
                    setup_type       = event_type,
                    strategy         = "structure_v1",
                    entry_price      = close,
                    stop_price       = c["stop_price"],
                    target_price     = c["target_price"],
                    r_multiple       = R_MULTIPLE,
                    status           = "active",
                    context_snapshot = json.dumps({
                        "event_type":   event_type,
                        "zone_center":  c["zone_center"],
                        "zone_touches": c["zone_touches"],
                        "vol_spike":    c["vol_spike"],
                        "vol_ratio":    c["vol_ratio"],
                        "trend":        c["trend"],
                        "ema_confirms": c["ema_confirms"],
                        "ema_20":       c["ema_20"],
                        "ema_50":       c["ema_50"],
                        "rsi_14":       c["rsi_14"],
                        "atr_14":       c["atr_14"],
                        "confidence":   c["confidence"],
                        "source":       source,
                    }),
                    created_at = datetime.now(UTC),
                ))
                signals_created += 1
                logger.info(
                    "signal | %s | %s | %s | zones=%d | conf=%.2f",
                    c["symbol"], event_type, direction,
                    c["zone_touches"], c["confidence"],
                )

        # ── 6. Finalise run record and commit ─────────────────────────────────
        run.finished_at           = datetime.now(UTC)
        run.scanned               = len(universe)
        run.candidates_considered = len(candidates)
        run.signals_created       = signals_created
        run.status                = "ok"
        db.commit()

        logger.info(
            "agent run done | signals=%d | candidates=%d | scan_time=%.0fms",
            signals_created, len(candidates), scan_time_ms,
        )

        return {
            "status":                "ok",
            "scanned":               len(universe),
            "candidates_considered": len(candidates),
            "signals_created":       signals_created,
            "zones_detected":        zones_detected,
            "scan_time_ms":          scan_time_ms,
        }

    except Exception as e:
        logger.exception("agent run failed: %s", e)
        db.rollback()
        db.add(AgentRun(
            started_at            = started_at,
            finished_at           = datetime.now(UTC),
            timeframe             = timeframe,
            source                = source,
            scanned               = 0,
            candidates_considered = 0,
            signals_created       = 0,
            status                = "error",
            error                 = str(e),
        ))
        db.commit()
        raise

    finally:
        db.close()


# ── Parallel worker ───────────────────────────────────────────────────────────

def _scan_one_symbol(
    symbol: str,
    *,
    timeframe: str,
    source: Optional[str],
    lookback: int,
) -> Optional[Dict[str, Any]]:
    """Fetch candles + run pipeline for one symbol in a dedicated DB session.

    Designed for ThreadPoolExecutor.  Creates and closes its own session so
    the main-thread session is never touched from a worker thread.
    Returns the pipeline result dict, or None if data is absent / insufficient.
    Exceptions are caught and logged so a single bad symbol never aborts the scan.
    """
    t0 = time.monotonic()
    db = SessionLocal()
    try:
        candles = _fetch_candles(db, symbol, timeframe, source, lookback)
        if not candles:
            return None

        result = run_structure_pipeline(
            candles,
            symbol    = symbol,
            timeframe = timeframe,
            source    = source,
        )
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        if result is not None:
            result["_scan_ms"] = elapsed_ms
            logger.debug(
                "%-6s | %-15s | zones=%-2d | vol=%-5s | trend=%-7s | conf=%.2f | %dms",
                symbol, result["event_type"], result["zones_detected"],
                result["vol_spike"], result["trend"],
                result["confidence"], elapsed_ms,
            )
        else:
            logger.debug("%-6s | no event | %dms", symbol, elapsed_ms)

        return result

    except Exception as exc:
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        logger.warning("%-6s | pipeline error | %dms | %s", symbol, elapsed_ms, exc)
        return None

    finally:
        db.close()


# ── DB helper ─────────────────────────────────────────────────────────────────

def _fetch_candles(
    db: Session,
    symbol: str,
    timeframe: str,
    source: Optional[str],
    lookback: int,
) -> List[Dict[str, Any]]:
    """Fetch and sort candle rows for one symbol; return list of dicts."""
    q = db.query(Candle).filter(
        Candle.symbol    == symbol,
        Candle.timeframe == timeframe,
    )
    if source:
        q = q.filter(Candle.source == source)

    items = q.order_by(Candle.ts.desc()).limit(lookback).all()
    if not items:
        return []

    items = sorted(items, key=lambda x: x.ts)
    return [
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the structure-first trading agent once")
    p.add_argument("--timeframe", default="1h",    help="Bar timeframe (default: 1h)")
    p.add_argument("--source",    default="yahoo", help="Data source filter (default: yahoo)")
    p.add_argument("--top-n",     type=int, default=10, dest="top_n",
                   help="Max candidates to evaluate (default: 10)")
    p.add_argument("--lookback",  type=int, default=200,
                   help="Bars of history per symbol (default: 200)")
    p.add_argument("--symbols",   default=None,
                   help="Comma-separated symbol override (default: major universe)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity (default: INFO)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else None
    )

    try:
        result = run_agent(
            timeframe = args.timeframe,
            source    = args.source,
            top_n     = args.top_n,
            lookback  = args.lookback,
            symbols   = symbols,
        )
    except Exception as e:
        print(f"[agent] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[agent] status:                {result['status']}")
    print(f"[agent] scanned:               {result['scanned']}")
    print(f"[agent] zones_detected:        {result['zones_detected']}")
    print(f"[agent] candidates_considered: {result['candidates_considered']}")
    print(f"[agent] signals_created:       {result['signals_created']}")
    print(f"[agent] scan_time_ms:          {result['scan_time_ms']}")

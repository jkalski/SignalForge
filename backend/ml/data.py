"""backend/ml/data.py

Backtest-to-training-data pipeline.

Runs the historical backtest across a symbol universe, extracts a flat
feature matrix from every completed setup (with a known outcome), and saves
the result as Parquet for the ML trainer.

Feature groups
--------------
  Signal context   — event_type, direction, signal_status
  Zone             — zone_touches, zones_ltf/htf_count, distance_pct
  Indicators       — vol_ratio, vol_spike, ema_confirms, atr_14, trend
  HTF / MTF        — htf_bias, mtf_aligned, near_htf_zone
  VWAP             — vwap_session_dist_pct, vwap_anchored_dist_pct
  Scoring          — confluence_score + per-dimension sub-scores
  Labels           — win (binary), return_pct, r_realized, outcome, bars_to_exit

Outputs (saved to out_prefix)
------------------------------
  <out_prefix>.parquet      primary ML input; one row per resolved setup
  <out_prefix>_meta.json    run metadata: counts, win rate, event breakdown

Usage
-----
    python -m backend.ml.data                           # full universe, 2y, 1h
    python -m backend.ml.data --symbols AAPL,MSFT,NVDA
    python -m backend.ml.data --period 1y --workers 4
    python -m backend.ml.data --out training_data/v2   # custom path prefix
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.backtest.runner import BacktestParams, run_backtest
from backend.marketdata.universe import SYMBOLS
from backend.marketdata.yahoo import fetch_yahoo_bars, YAHOO_INTERVAL_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEFRAME = "1h"
_DEFAULT_PERIOD    = "2y"   # yfinance: 1h allows up to 730 days (2y)
_DEFAULT_OUT       = Path("training_data") / "features"
_WORKERS           = 6

# HTF timeframe map — only yfinance-supported intervals.
# 4h isn't available on yfinance so 1h→1d (daily gives ample zone context).
_HTF_MAP: Dict[str, str] = {
    "1m":  "15m",
    "5m":  "1h",
    "15m": "1h",
    "1h":  "1d",
    "4h":  "1d",
    "1d":  "1d",
}

# Column order: metadata → raw features → sub-scores → labels
_COL_ORDER = [
    "symbol", "timeframe", "ts",
    "event_type", "direction", "signal_status",
    "zone_touches", "zones_ltf_count", "zones_htf_count", "distance_pct",
    "vol_ratio", "vol_spike", "ema_confirms", "atr_14", "trend",
    "htf_bias", "mtf_aligned", "near_htf_zone",
    "vwap_session_dist_pct", "vwap_anchored_dist_pct",
    "confluence_score",
    "score_zone", "score_volume", "score_event",
    "score_ema", "score_vwap", "score_mtf", "score_divergence",
    "outcome", "win", "return_pct", "r_realized", "bars_to_exit",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _sub_score(reasons: Dict[str, Any], dim: str) -> int:
    """Pull the score integer from one confluence_reasons dimension."""
    d = reasons.get(dim)
    return int(d.get("score", 0)) if isinstance(d, dict) else 0


def extract_features(
    setup: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """
    Flatten one backtest setup dict into a single training row.

    Returns None when the outcome isn't resolved (return_pct is None),
    meaning the bar was too close to the end of the candle window for the
    outcome horizon to complete.
    """
    if setup.get("return_pct") is None:
        return None

    reasons: Dict[str, Any] = setup.get("confluence_reasons") or {}

    def _float(key: str) -> float:
        v = setup.get(key)
        return float(v) if v is not None else float("nan")

    return {
        # ── Metadata ──────────────────────────────────────────────────────
        "symbol":    symbol,
        "timeframe": timeframe,
        "ts":        setup["ts"],

        # ── Signal context ────────────────────────────────────────────────
        "event_type":    setup.get("event_type", ""),
        "direction":     setup.get("direction", ""),
        "signal_status": setup.get("signal_status", ""),

        # ── Zone ──────────────────────────────────────────────────────────
        "zone_touches":    int(setup.get("zone_touches") or 0),
        "zones_ltf_count": int(setup.get("zones_ltf_count") or 0),
        "zones_htf_count": int(setup.get("zones_htf_count") or 0),
        "distance_pct":    float(setup.get("distance_pct") or 0.0),

        # ── Indicators ────────────────────────────────────────────────────
        "vol_ratio":    _float("vol_ratio"),
        "vol_spike":    int(bool(setup.get("vol_spike", False))),
        "ema_confirms": int(bool(setup.get("ema_confirms", False))),
        "atr_14":       float(setup.get("atr_14") or 0.0),
        "trend":        setup.get("trend", "neutral") or "neutral",

        # ── HTF / MTF ─────────────────────────────────────────────────────
        "htf_bias":    setup.get("htf_bias", "neutral") or "neutral",
        "mtf_aligned": int(bool(setup.get("mtf_aligned", False))),
        "near_htf_zone": int(bool(setup.get("near_htf_zone", False))),

        # ── VWAP ──────────────────────────────────────────────────────────
        "vwap_session_dist_pct":  _float("vwap_session_dist_pct"),
        "vwap_anchored_dist_pct": _float("vwap_anchored_dist_pct"),

        # ── Scoring ───────────────────────────────────────────────────────
        "confluence_score": int(setup.get("confluence_score") or 0),
        "score_zone":       _sub_score(reasons, "zone"),
        "score_volume":     _sub_score(reasons, "volume"),
        "score_event":      _sub_score(reasons, "event"),
        "score_ema":        _sub_score(reasons, "ema"),
        "score_vwap":       _sub_score(reasons, "vwap"),
        "score_mtf":        _sub_score(reasons, "mtf"),
        "score_divergence": _sub_score(reasons, "divergence"),

        # ── Labels ────────────────────────────────────────────────────────
        "outcome":      setup.get("outcome") or "timeout",
        "win":          int(setup.get("outcome") == "win"),
        "return_pct":   float(setup.get("return_pct") or 0.0),
        "r_realized":   _float("r_realized"),
        "bars_to_exit": int(setup["bars_to_exit"]) if setup.get("bars_to_exit") is not None else -1,
    }


# ---------------------------------------------------------------------------
# Per-symbol worker
# ---------------------------------------------------------------------------


def _process_symbol(
    symbol: str,
    timeframe: str,
    period: str,
) -> List[Dict[str, Any]]:
    """
    Fetch candles, run the full backtest, extract feature rows.

    Designed for ThreadPoolExecutor — returns an empty list on any error
    so a single bad symbol never aborts the run.
    """
    t0 = time.monotonic()
    htf_tf = _HTF_MAP.get(timeframe)

    try:
        ltf_candles = fetch_yahoo_bars(symbol, timeframe, period=period)
        if len(ltf_candles) < 100:
            logger.warning("%-6s | only %d LTF bars — skipping", symbol, len(ltf_candles))
            return []

        # HTF candles — optional; degrade gracefully if timeframe unsupported.
        htf_candles: List[Dict] = []
        if htf_tf and htf_tf in YAHOO_INTERVAL_MAP:
            try:
                htf_candles = fetch_yahoo_bars(symbol, htf_tf, period=period)
            except Exception as e:
                logger.debug("%-6s | HTF fetch failed (%s): %s", symbol, htf_tf, e)

        setups, _ = run_backtest(
            ltf_candles,
            params=BacktestParams(
                min_confluence_score=45,  # only meaningful setups; cuts noise significantly
                only_active=True,         # require vol_spike + ema_confirms + mtf_aligned
            ),
            htf_candles=htf_candles or None,
            symbol=symbol,
            timeframe=timeframe,
        )

        rows = [
            row for s in setups
            if (row := extract_features(s, symbol, timeframe)) is not None
        ]

        elapsed = round(time.monotonic() - t0, 1)
        logger.info(
            "%-6s | bars=%-4d | setups=%-3d | rows=%-3d | %.1fs",
            symbol, len(ltf_candles), len(setups), len(rows), elapsed,
        )
        return rows

    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 1)
        logger.warning("%-6s | ERROR %.1fs | %s", symbol, elapsed, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_data(
    symbols:    Optional[List[str]] = None,
    timeframe:  str                 = _DEFAULT_TIMEFRAME,
    period:     str                 = _DEFAULT_PERIOD,
    out_prefix: str | Path          = _DEFAULT_OUT,
    workers:    int                 = _WORKERS,
) -> pd.DataFrame:
    """
    Run the backtest across all symbols and save a labeled feature DataFrame.

    Parameters
    ----------
    symbols : list or None
        Symbols to process.  None → full universe from universe.py.
    timeframe : str
        Bar timeframe (e.g. "1h", "1d").  Must be in YAHOO_INTERVAL_MAP.
    period : str
        yfinance period string ("2y", "1y", "6mo", …).
        For "1h", yfinance allows a maximum of "2y" (~730 days).
    out_prefix : str or Path
        File path prefix.  ".parquet" and "_meta.json" are appended.
    workers : int
        Parallel download threads.  Keep ≤ 8 to avoid yfinance rate limits.

    Returns
    -------
    pd.DataFrame
        Feature matrix.  Empty DataFrame if no data was produced.
    """
    universe   = symbols or SYMBOLS
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building training data | symbols=%d | timeframe=%s | period=%s",
        len(universe), timeframe, period,
    )
    t_start = time.monotonic()

    all_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_symbol, sym, timeframe, period): sym
            for sym in universe
        }
        for fut in concurrent.futures.as_completed(futures):
            all_rows.extend(fut.result())

    if not all_rows:
        logger.warning("No training rows produced — check symbols and candle availability")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Apply consistent column ordering; drop any extras.
    present = [c for c in _COL_ORDER if c in df.columns]
    df = df[present]

    # Save parquet
    parquet_path = out_prefix.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_path)

    # Save meta
    win_rate = float(df["win"].mean()) if "win" in df.columns else None
    meta = {
        "symbols":       universe,
        "timeframe":     timeframe,
        "period":        period,
        "total_rows":    len(df),
        "symbols_count": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
        "win_rate":      round(win_rate, 4) if win_rate is not None else None,
        "event_counts":  df["event_type"].value_counts().to_dict() if "event_type" in df.columns else {},
        "outcome_counts": df["outcome"].value_counts().to_dict() if "outcome" in df.columns else {},
        "elapsed_s":     round(time.monotonic() - t_start, 1),
    }
    meta_path = out_prefix.parent / (out_prefix.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Meta → %s", meta_path)

    logger.info(
        "Done | rows=%d | symbols=%d | win_rate=%.1f%% | %.0fs",
        meta["total_rows"],
        meta["symbols_count"],
        (win_rate or 0) * 100,
        meta["elapsed_s"],
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ML training data from backtest history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols (default: full universe)",
    )
    p.add_argument(
        "--timeframe", default=_DEFAULT_TIMEFRAME,
        help="Bar timeframe",
    )
    p.add_argument(
        "--period", default=_DEFAULT_PERIOD,
        help="yfinance period string (1h max is '2y')",
    )
    p.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help="Output path prefix (.parquet and _meta.json appended)",
    )
    p.add_argument(
        "--workers", type=int, default=_WORKERS,
        help="Parallel download threads (keep ≤ 8)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else None
    )

    try:
        df = build_training_data(
            symbols   = symbols,
            timeframe = args.timeframe,
            period    = args.period,
            out_prefix = args.out,
            workers   = args.workers,
        )
    except Exception as e:
        print(f"[ml.data] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("[ml.data] No data produced — check logs above.")
        sys.exit(1)

    print(f"\n[ml.data] Training dataset built")
    print(f"  rows:        {len(df):,}")
    print(f"  symbols:     {df['symbol'].nunique()}")
    print(f"  win rate:    {df['win'].mean():.1%}")
    print(f"  event types: {df['event_type'].nunique()}")
    print(f"\nEvent type breakdown (count | win rate):")
    summary = (
        df.groupby("event_type")["win"]
        .agg(count="count", win_rate="mean")
        .sort_values("count", ascending=False)
    )
    summary["win_rate"] = summary["win_rate"].map("{:.1%}".format)
    print(summary.to_string())

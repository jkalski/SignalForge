"""backend/api/routes/analyze.py

On-demand ticker analysis endpoint.

GET /analyze/{symbol}
    Run the full signal pipeline against live yfinance data and return
    a probability estimate + plain-English explanation.

GET /analyze/{symbol}/proximity
    Check whether price is approaching a key zone even if no signal has
    fired yet — the "entering an area" detection.

Examples
--------
    GET /analyze/AAPL
    GET /analyze/MSFT?timeframe=1h
    GET /analyze/SPY/proximity
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.agent.pipeline import run_structure_pipeline
from backend.indicators.pivots import find_pivots
from backend.indicators.zones import build_zones_from_pivots
from backend.marketdata.yahoo import YAHOO_INTERVAL_MAP, fetch_yahoo_bars
from backend.signals.explain import format_alert, format_no_signal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# LTF → HTF map (yfinance-supported intervals only)
_HTF_MAP: Dict[str, str] = {
    "1m":  "15m",
    "5m":  "1h",
    "15m": "1h",
    "1h":  "1d",
    "4h":  "1d",
    "1d":  "1d",
}

_LOOKBACK     = 200   # bars fed to the pipeline (matches live agent)
_PROXIMITY_PCT = 1.5  # % distance to nearest zone to flag as "approaching"


# ---------------------------------------------------------------------------
# GET /analyze/{symbol}
# ---------------------------------------------------------------------------


@router.get("/{symbol}")
def analyze_symbol(
    symbol:    str,
    timeframe: str = Query(default="1h", description="Bar timeframe (1m/5m/15m/1h/1d)"),
) -> Dict[str, Any]:
    """
    Run the full pipeline on live data and return probability + explanation.

    When a signal is detected the response includes:
      - probability       0.0–1.0 (ML model if trained, else confluence/100)
      - probability_pct   human-readable e.g. "67%"
      - explanation       plain-English 2-3 sentence description
      - key_factors       bullet-point list of top confirmations
      - price             entry / stop / target levels
      - signal_status     "active" | "watchlist"

    When no signal fires:
      - signal_status     "no_signal"
      - explanation       why nothing triggered
    """
    symbol = symbol.upper().strip()

    if timeframe not in YAHOO_INTERVAL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe '{timeframe}'. "
                   f"Choose from: {sorted(YAHOO_INTERVAL_MAP.keys())}",
        )

    # ── Fetch candles ─────────────────────────────────────────────────────
    try:
        ltf_candles = fetch_yahoo_bars(symbol, timeframe, period="1y" if timeframe == "1d" else "6mo")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {symbol}: {e}")

    if len(ltf_candles) < 60:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient data for {symbol} — only {len(ltf_candles)} bars returned.",
        )

    # Use the most recent lookback bars
    ltf_candles = ltf_candles[-_LOOKBACK:]

    # Optional HTF enrichment
    htf_candles: List[Dict] = []
    htf_tf = _HTF_MAP.get(timeframe)
    if htf_tf and htf_tf != timeframe:
        try:
            htf_candles = fetch_yahoo_bars(symbol, htf_tf, period="2y")
        except Exception:
            pass  # HTF optional

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        result = run_structure_pipeline(
            ltf_candles,
            symbol      = symbol,
            timeframe   = timeframe,
            source      = "yahoo",
            htf_candles = htf_candles or None,
        )
    except Exception as e:
        logger.exception("Pipeline error for %s: %s", symbol, e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # ── Format response ───────────────────────────────────────────────────
    if result is None:
        return format_no_signal(symbol, timeframe)

    return format_alert(result)


# ---------------------------------------------------------------------------
# GET /analyze/{symbol}/proximity
# ---------------------------------------------------------------------------


@router.get("/{symbol}/proximity")
def analyze_proximity(
    symbol:    str,
    timeframe: str = Query(default="1h"),
) -> Dict[str, Any]:
    """
    Check whether price is approaching a key structural zone, even if no
    signal has fired yet.

    Returns the nearest support and resistance zones, the distance to each,
    and whether price is within the proximity threshold (default 1.5%).
    """
    symbol = symbol.upper().strip()

    if timeframe not in YAHOO_INTERVAL_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe '{timeframe}'.")

    try:
        candles = fetch_yahoo_bars(symbol, timeframe, period="6mo")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {symbol}: {e}")

    if len(candles) < 60:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}.")

    candles = candles[-_LOOKBACK:]
    close   = float(candles[-1]["close"])

    # Build zones using the same pivot/zone logic as the live pipeline
    try:
        import pandas as pd
        df        = pd.DataFrame(candles)
        df["ts"]  = pd.to_datetime(df["ts"])
        atr_series = (
            df["close"].diff().abs()
            .ewm(alpha=1/14, adjust=False).mean()
        )
        atr       = float(atr_series.iloc[-1])
        zone_tol  = atr * 0.5

        df_piv  = find_pivots(df, order=3)
        zones   = build_zones_from_pivots(df_piv, zone_tol=zone_tol, min_touches=2)
    except Exception as e:
        logger.warning("Zone detection failed for %s: %s", symbol, e)
        zones = []

    if not zones:
        return {
            "symbol":    symbol,
            "timeframe": timeframe,
            "close":     close,
            "zones":     [],
            "approaching": False,
            "message":   f"No structural zones detected for {symbol} on {timeframe}.",
        }

    # Classify zones as support or resistance relative to current price
    supports    = [z for z in zones if z.center < close]
    resistances = [z for z in zones if z.center >= close]

    nearest_support    = max(supports,    key=lambda z: z.center) if supports    else None
    nearest_resistance = min(resistances, key=lambda z: z.center) if resistances else None

    def _zone_info(zone) -> Optional[Dict]:
        if zone is None:
            return None
        dist_pct = abs(close - zone.center) / close * 100
        return {
            "center":    round(zone.center, 4),
            "touches":   zone.touches,
            "dist_pct":  round(dist_pct, 2),
            "approaching": dist_pct <= _PROXIMITY_PCT,
        }

    support_info    = _zone_info(nearest_support)
    resistance_info = _zone_info(nearest_resistance)

    approaching = bool(
        (support_info    and support_info["approaching"])
        or (resistance_info and resistance_info["approaching"])
    )

    # Build message
    messages = []
    if support_info and support_info["approaching"]:
        messages.append(
            f"{symbol} is within {support_info['dist_pct']:.1f}% of a "
            f"{support_info['touches']}-touch support zone at ${support_info['center']:,.2f}"
        )
    if resistance_info and resistance_info["approaching"]:
        messages.append(
            f"{symbol} is within {resistance_info['dist_pct']:.1f}% of a "
            f"{resistance_info['touches']}-touch resistance zone at ${resistance_info['center']:,.2f}"
        )

    return {
        "symbol":          symbol,
        "timeframe":       timeframe,
        "close":           round(close, 4),
        "approaching":     approaching,
        "message":         " | ".join(messages) if messages else f"No key zones within {_PROXIMITY_PCT}% of current price.",
        "nearest_support":    support_info,
        "nearest_resistance": resistance_info,
    }

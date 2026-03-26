"""
backend/backtest/runner.py

Historical backtest of the structure-first signal pipeline.

Replays candles bar-by-bar through the EXACT same sub-functions used by the
live pipeline.  No logic is duplicated — all imports come from the shared
indicator / signal modules.  Only orchestration lives here.

Performance strategy
--------------------
EWM indicators (ATR14, EMA50) and vol_sma are pre-computed on the full series
ONCE.  Because pandas EWM with adjust=False is a recursive formula
(value[i] depends only on values[0..i]), computing on the full series produces
bit-for-bit identical values to computing on every prefix slice individually —
zero look-ahead bias.

find_pivots + build_zones DO require a growing window per bar (pivot detection
needs future bars to confirm a local extremum), so they run inside the loop.

HTF enrichment
--------------
If htf_candles is provided, the HTF slice at each LTF bar i includes only
candles with ts <= ltf_ts[i], ensuring correct temporal alignment.

Output
------
run_backtest() → (setups: list[dict], summary: dict)

save_results(setups, summary, out_path) writes:
  <out_path>.csv           — one row per triggered setup
  <out_path>_summary.json  — aggregate metrics and params

CLI usage
---------
    python -m backend.backtest.runner \\
        --symbol AAPL --timeframe 1h --source yahoo --lookback 500 \\
        --out results/AAPL_1h

    # parameter sweep example:
    python -m backend.backtest.runner --symbol MSFT --pivot-order 5 \\
        --zone-atr-mult 0.3 --vol-spike-mult 2.5 --min-score 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Shared sub-functions — identical to live pipeline imports ─────────────────
from backend.features.indicators import compute_features
from backend.indicators.pivots import find_pivots
from backend.indicators.trend import add_trend_filter
from backend.indicators.zones import Zone, build_zones_from_pivots
from backend.indicators.vwap import (
    add_session_vwap,
    add_anchored_vwap,
    get_last_major_pivot_index,
)
from backend.indicators.mtf import (
    align_zones,
    get_htf_bias,
    filter_signal_by_htf,
    _near_any_htf_zone,
)
from backend.signals.structure_signals import detect_breakout_or_bounce
from backend.signals.liquidity import detect_liquidity_sweeps
from backend.signals.scoring import score_setup
from backend.indicators.daily_levels import get_daily_levels
from backend.signals.fvg import detect_fvg
from backend.signals.vwap_signals import detect_vwap_reclaim
from backend.signals.orb import detect_orb
from backend.signals.market_structure import detect_bos
from backend.signals.divergence import detect_rsi_divergence
from backend.signals.gap import detect_gap
from backend.signals.bar_patterns import detect_inside_bar, detect_outside_bar, detect_ath_breakout

# Private helpers re-used from the live pipeline (pure, stateless).
# Importing with underscore names is valid Python; nothing is duplicated.
from backend.agent.pipeline import (
    _select_nearby,
    _find_zone_by_center,
    _DIRECTION,
    _MIN_BARS,
    _MIN_HTF_BARS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class BacktestParams:
    """
    All tunable knobs for a backtest run.

    Defaults match the live pipeline constants so that ``BacktestParams()``
    reproduces the exact live behaviour without any adjustments.
    """

    # ── Structural indicators ─────────────────────────────────────────────
    pivot_order:      int   = 3     # swing detection half-window (± bars)
    zone_atr_mult:    float = 0.5   # zone_tol = ATR14 × this
    min_zone_touches: int   = 2     # discard zones with fewer pivot touches
    max_nearby_zones: int   = 6     # zones evaluated by event detectors
    sweep_lookback:   int   = 2     # bars back used for sweep wick search

    # ── Volume ────────────────────────────────────────────────────────────
    vol_spike_mult:   float = 1.8   # spike flag = volume > vol_sma × this

    # ── Position sizing ───────────────────────────────────────────────────
    atr_stop_mult:    float = 1.5   # stop  = close ± ATR × this
    atr_target_mult:  float = 3.0   # target = close ± ATR × this

    # ── Outcome evaluation ────────────────────────────────────────────────
    outcome_horizon_bars: int = 20  # bars forward to check stop/target

    # ── Filters ───────────────────────────────────────────────────────────
    min_confluence_score: int  = 0     # skip setups below this score
    only_active:          bool = False  # if True, skip watchlist setups
    event_types: Optional[List[str]] = None  # None = all; list = allow-list filter

    # ── Zone method ───────────────────────────────────────────────────────
    zone_method: str = "pivots"  # "pivots" | "supply_demand"

    # ── Regime filter ─────────────────────────────────────────────────────
    regime_filter: bool = False   # if True, gate direction by regime_symbol EMA50
    regime_symbol: str  = "SPY"   # symbol used as regime indicator

    # ── Zone lookback ─────────────────────────────────────────────────────
    # Caps the rolling window fed to find_pivots / build_zones each bar.
    # Matches the live pipeline's lookback (200 bars) and turns O(n²)
    # pivot detection into O(n × zone_lookback), making long runs ~17× faster.
    # Set to 0 to disable the cap (original expanding-window behaviour).
    zone_lookback: int = 200

    # ── HTF ───────────────────────────────────────────────────────────────
    disable_htf: bool = False  # force no HTF enrichment even if candles given

    # ── Execution mode ────────────────────────────────────────────────────
    entry_mode:   str   = "enter_on_close"   # "enter_on_close" | "enter_on_next_open"
    slippage_bps: float = 0.0                # basis-points slippage added to entry price

    @property
    def r_multiple(self) -> float:
        return round(self.atr_target_mult / self.atr_stop_mult, 2)

    @property
    def min_bars(self) -> int:
        return _MIN_BARS  # 52 (EMA50 needs 50+ bars)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    candles: List[Dict[str, Any]],
    params: Optional[BacktestParams] = None,
    htf_candles: Optional[List[Dict[str, Any]]] = None,
    regime_candles: Optional[List[Dict[str, Any]]] = None,
    symbol: str = "",
    timeframe: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run the structure-first pipeline bar-by-bar on a historical candle series.

    Parameters
    ----------
    candles : list of dicts
        Full OHLCV history sorted ascending by ts.
        Required keys: ts, open, high, low, close, volume.
        volume may be None/0 — all volume-dependent steps degrade gracefully.
    params : BacktestParams or None
        Tunable parameters.  None → defaults reproduce live behaviour exactly.
    htf_candles : list of dicts or None
        Higher-timeframe OHLCV (same schema, same ts ordering).
        Sliced up to each LTF bar's ts for correct temporal alignment.
    symbol, timeframe : str
        Metadata attached to every setup record and the summary dict.

    Returns
    -------
    setups : list of dicts
        One record per triggered bar that passed the score / status filters.
        Outcome fields (outcome, exit_bar, return_pct, r_realized, …) are
        populated by looking forward ``params.outcome_horizon_bars`` bars.
    summary : dict
        Aggregate statistics: win_rate, avg_r, max_drawdown_r,
        false_breakout_rate, signal_frequency_per_100, by_event_type, params.
    """
    if params is None:
        params = BacktestParams()

    if len(candles) < params.min_bars:
        logger.warning(
            "backtest | %s | too few candles: %d < %d",
            symbol, len(candles), params.min_bars,
        )
        return [], {
            "symbol": symbol, "timeframe": timeframe,
            "error": f"Need at least {params.min_bars} candles, got {len(candles)}",
        }

    # ── 1. Build full DataFrame ────────────────────────────────────────────
    candles = sorted(candles, key=lambda x: x["ts"])
    df = pd.DataFrame(candles)
    df["ts"] = pd.to_datetime(df["ts"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    # ── 2. Pre-compute backward-looking indicators on the full series ──────
    #
    # EWM(adjust=False) is a recursive formula: EMA_t = α·close_t + (1−α)·EMA_{t-1}
    # Computing on the full series is mathematically identical to computing
    # on every prefix separately — no look-ahead.  Same holds for rolling SMA.
    #
    close = df["close"]
    hi    = df["high"]
    lo    = df["low"]
    prev  = close.shift(1)
    tr    = pd.concat(
        [(hi - lo), (hi - prev).abs(), (lo - prev).abs()], axis=1
    ).max(axis=1)

    df["atr_14"]    = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["ema50"]     = close.ewm(span=50, adjust=False).mean()
    df["trend_bull"] = close > df["ema50"]
    df["trend_bear"] = close < df["ema50"]
    df["vol_sma"]   = df["volume"].rolling(20, min_periods=20).mean()

    # Pre-compute cumulative valid-volume count for O(1) volume_ok per bar.
    vol_valid_cum = (df["volume"].fillna(0) > 0).astype(int).cumsum().to_numpy()

    # ── 3. Prepare HTF candles ─────────────────────────────────────────────
    htf_sorted: Optional[List[Dict]] = None
    htf_ts_ns:  Optional[np.ndarray] = None

    if not params.disable_htf and htf_candles and len(htf_candles) >= _MIN_HTF_BARS:
        htf_sorted = sorted(htf_candles, key=lambda x: x["ts"])
        htf_ts_ns  = np.array(
            [pd.Timestamp(c["ts"]).value for c in htf_sorted],
            dtype=np.int64,
        )

    # ── 3b. Prepare regime candles (SPY or other) ─────────────────────────
    spy_sorted:  Optional[List[Dict]] = None
    spy_ts_ns:   Optional[np.ndarray] = None
    spy_ema50:   Optional[np.ndarray] = None
    spy_closes:  Optional[np.ndarray] = None

    if params.regime_filter and regime_candles and len(regime_candles) >= 50:
        spy_sorted = sorted(regime_candles, key=lambda x: x["ts"])
        spy_ts_ns  = np.array(
            [pd.Timestamp(c["ts"]).value for c in spy_sorted],
            dtype=np.int64,
        )
        spy_df            = pd.DataFrame(spy_sorted)
        spy_df["close"]   = spy_df["close"].astype(float)
        spy_df["ema50"]   = spy_df["close"].ewm(span=50, adjust=False).mean()
        spy_ema50         = spy_df["ema50"].to_numpy(dtype=np.float64)
        spy_closes        = spy_df["close"].to_numpy(dtype=np.float64)

    # ── 4. OHLCV-only view for window slicing ─────────────────────────────
    # find_pivots / zone detection should see only raw OHLCV, not the
    # pre-computed columns added above.
    ohlcv_cols = [c for c in ("ts", "open", "high", "low", "close", "volume")
                  if c in df.columns]
    df_ohlcv   = df[ohlcv_cols]

    # ── 5. Bar-by-bar scan ─────────────────────────────────────────────────
    setups: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    logger.info(
        "backtest | %s %s | bars=%d | %s → %s",
        symbol or "?", timeframe or "?",
        len(df), df["ts"].iloc[0], df["ts"].iloc[-1],
    )

    for i in range(params.min_bars - 1, len(df)):
        feat    = df.iloc[i]
        atr     = float(feat["atr_14"])
        close_i = float(feat["close"])

        if not (math.isfinite(atr) and atr > 0):
            continue
        if not (math.isfinite(close_i) and close_i > 0):
            continue

        # Volume quality: O(1) using pre-computed cumulative sum.
        volume_ok = int(vol_valid_cum[i]) >= int((i + 1) * 0.80)

        # Custom vol_spike (tunable multiplier, not the hardcoded 1.8×).
        vol_sma_i = float(feat["vol_sma"]) if pd.notna(feat["vol_sma"]) else None
        vol_i     = float(feat["volume"])  if pd.notna(feat["volume"])  else None
        vol_spike = bool(
            vol_i and vol_sma_i and vol_sma_i > 0
            and vol_i > vol_sma_i * params.vol_spike_mult
        )
        vol_ratio = (
            round(vol_i / vol_sma_i, 2)
            if (vol_i and vol_sma_i and vol_sma_i > 0)
            else None
        )

        # Trend bias (pre-computed; no re-computation needed).
        trend_bull = bool(feat["trend_bull"])
        trend_bear = bool(feat["trend_bear"])
        trend      = "bull" if trend_bull else "bear" if trend_bear else "neutral"

        # Slice HTF up to current LTF timestamp (binary search, O(log n)).
        htf_slice: Optional[List[Dict]] = None
        if htf_sorted is not None:
            ltf_ts_val = df["ts"].iloc[i].value
            htf_end    = int(np.searchsorted(htf_ts_ns, ltf_ts_val, side="right"))
            if htf_end >= _MIN_HTF_BARS:
                htf_slice = htf_sorted[:htf_end]

        # Regime filter: look up SPY EMA50 at this bar (binary search, O(log n)).
        regime_bias: Optional[str] = None
        if spy_sorted is not None:
            ltf_ts_val  = df["ts"].iloc[i].value
            spy_idx     = int(np.searchsorted(spy_ts_ns, ltf_ts_val, side="right")) - 1
            if spy_idx >= 49:  # EMA50 needs 50 bars to warm up
                sc = spy_closes[spy_idx]
                se = spy_ema50[spy_idx]
                if np.isfinite(sc) and np.isfinite(se) and se > 0:
                    regime_bias = "bullish" if sc > se else "bearish" if sc < se else "neutral"

        # Core per-bar analysis (pivots, zones, events, scoring).
        # Cap the zone-detection window to zone_lookback bars so complexity
        # stays O(n × zone_lookback) rather than O(n²).  Matches live behaviour
        # where the agent fetches only the last `lookback` bars from the DB.
        win_start = (
            max(0, i + 1 - params.zone_lookback)
            if params.zone_lookback > 0 else 0
        )
        result = _run_bar(
            window     = df_ohlcv.iloc[win_start: i + 1],
            atr        = atr,
            close_i    = close_i,
            vol_spike  = vol_spike,
            vol_ratio  = vol_ratio,
            volume_ok  = volume_ok,
            trend      = trend,
            trend_bull = trend_bull,
            trend_bear = trend_bear,
            bar_ts     = df["ts"].iloc[i],
            params     = params,
            htf_slice  = htf_slice,
        )
        if result is None:
            continue
        if result["confluence_score"] < params.min_confluence_score:
            continue
        if params.only_active and result["signal_status"] != "active":
            continue
        if params.event_types and result["event_type"] not in params.event_types:
            continue
        if regime_bias is not None:
            if regime_bias == "bearish" and result["direction"] == "long":
                continue
            if regime_bias == "bullish" and result["direction"] == "short":
                continue

        result["regime_bias"] = regime_bias
        result["bar_index"] = i
        result["symbol"]    = symbol
        result["timeframe"] = timeframe
        setups.append(result)

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("backtest | done | setups=%d | %.0fms", len(setups), elapsed_ms)

    # ── 6. Resolve outcomes (look-forward, post-scan) ──────────────────────
    _resolve_outcomes(setups, df, params)

    # ── 7. Aggregate summary ───────────────────────────────────────────────
    summary = _compute_summary(setups, len(df), params)
    summary.update({"symbol": symbol, "timeframe": timeframe, "scan_time_ms": elapsed_ms})
    return setups, summary


def save_results(
    setups: List[Dict[str, Any]],
    summary: Dict[str, Any],
    out_path: str,
) -> None:
    """
    Write backtest results to disk.

    Creates two files:
      <out_path>.csv           — one row per setup; nested dicts JSON-encoded
      <out_path>_summary.json  — aggregate metrics and params

    Parameters
    ----------
    out_path : str
        Path stem without extension (e.g. ``"results/AAPL_1h"``).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    csv_path  = p.with_suffix(".csv")
    json_path = p.parent / (p.stem + "_summary.json")

    if setups:
        # Flatten: all scalar keys first, then nested confluence_reasons last.
        nested_keys = {"confluence_reasons"}
        flat_keys   = [k for k in setups[0] if k not in nested_keys]
        fieldnames  = flat_keys + list(nested_keys)

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in setups:
                row = {
                    **s,
                    "ts":                  str(s.get("ts", "")),
                    "exit_ts":             str(s.get("exit_ts", "")) if s.get("exit_ts") is not None else "",
                    "confluence_reasons":  json.dumps(s.get("confluence_reasons")),
                }
                writer.writerow(row)
        logger.info("saved %d setups → %s", len(setups), csv_path)
    else:
        logger.info("no setups to write — CSV skipped")

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("saved summary → %s", json_path)


# ---------------------------------------------------------------------------
# Per-bar analysis (mirrors run_structure_pipeline, fully parameterized)
# ---------------------------------------------------------------------------


def _run_bar(
    window:     pd.DataFrame,
    atr:        float,
    close_i:    float,
    vol_spike:  bool,
    vol_ratio:  Optional[float],
    volume_ok:  bool,
    trend:      str,
    trend_bull: bool,
    trend_bear: bool,
    bar_ts:     pd.Timestamp,
    params:     BacktestParams,
    htf_slice:  Optional[List[Dict]],
) -> Optional[Dict[str, Any]]:
    """
    Run the full analysis pipeline on one expanding candle window.

    Every call below mirrors the corresponding step in run_structure_pipeline.
    The step numbering matches the live pipeline docstring intentionally.
    """
    zone_tol = atr * params.zone_atr_mult

    # ── Step 4: pivots on the expanding window ────────────────────────────
    window_piv = find_pivots(window, order=params.pivot_order)

    # Look-ahead guard — pivot confirmation delay:
    # argrelextrema marks a pivot at row j only when it is the strict
    # extremum across [j-order, j+order].  The last `pivot_order` rows of
    # the expanding window have no right-side confirmation bars — any flag
    # there would be based on future data unavailable at bar i.  Zero them
    # out before zone clustering so zones are built only from confirmed pivots.
    if params.pivot_order > 0 and len(window_piv) > params.pivot_order:
        window_piv = window_piv.copy()
        ph_col = window_piv.columns.get_loc("pivot_high")
        pl_col = window_piv.columns.get_loc("pivot_low")
        window_piv.iloc[-params.pivot_order:, ph_col] = False
        window_piv.iloc[-params.pivot_order:, pl_col] = False

    # ── Step 7: LTF zone clustering ───────────────────────────────────────
    if params.zone_method == "supply_demand":
        from backend.indicators.sd_zones import build_sd_zones
        ltf_zones = build_sd_zones(window_piv, zone_tol=zone_tol, atr=atr)
    else:
        ltf_zones = build_zones_from_pivots(
            window_piv, zone_tol=zone_tol, min_touches=params.min_zone_touches
        )
    if not ltf_zones:
        return None

    # ── Step 7.5: merge PDH/PDL zones ─────────────────────────────────────
    pdh_pdl = get_daily_levels(window_piv, zone_tol)
    if pdh_pdl:
        ltf_zones = ltf_zones + pdh_pdl

    # ── Step 8: HTF zones + bias (optional) ──────────────────────────────
    htf_zones: List[Zone] = []
    htf_bias  = "neutral"
    htf_atr   = atr
    alignment: Dict = {
        "linked": [], "unlinked_ltf": list(ltf_zones), "htf_only": []
    }
    near_htf = False

    if htf_slice and len(htf_slice) >= _MIN_HTF_BARS:
        htf_df = pd.DataFrame(htf_slice)
        htf_df["ts"] = pd.to_datetime(htf_df["ts"])
        for col in ("open", "high", "low", "close"):
            htf_df[col] = htf_df[col].astype(float)
        if "volume" in htf_df.columns:
            htf_df["volume"] = pd.to_numeric(htf_df["volume"], errors="coerce")
        else:
            htf_df["volume"] = np.nan

        htf_rows = compute_features(htf_slice)
        _raw_htf_atr = htf_rows[-1].get("atr_14")
        if _raw_htf_atr and math.isfinite(_raw_htf_atr) and _raw_htf_atr > 0:
            htf_atr = _raw_htf_atr

        htf_df = find_pivots(htf_df, order=params.pivot_order)
        # Same confirmation delay on the HTF series — the last pivot_order
        # HTF bars cannot yet be confirmed by future HTF closes.
        if params.pivot_order > 0 and len(htf_df) > params.pivot_order:
            ph_col = htf_df.columns.get_loc("pivot_high")
            pl_col = htf_df.columns.get_loc("pivot_low")
            htf_df.iloc[-params.pivot_order:, ph_col] = False
            htf_df.iloc[-params.pivot_order:, pl_col] = False
        htf_df    = add_trend_filter(htf_df)
        htf_tol   = htf_atr * params.zone_atr_mult
        if params.zone_method == "supply_demand":
            from backend.indicators.sd_zones import build_sd_zones
            htf_zones = build_sd_zones(htf_df, zone_tol=htf_tol, atr=htf_atr)
        else:
            htf_zones = build_zones_from_pivots(
                htf_df, zone_tol=htf_tol, min_touches=params.min_zone_touches
            )
        htf_bias  = get_htf_bias(htf_df)

        if htf_zones:
            mtf_tol   = max(htf_tol, zone_tol)
            alignment = align_zones(htf_zones, ltf_zones, tol=mtf_tol)

    # ── Step 9: VWAP (volume-gated) ───────────────────────────────────────
    vwap_session_dist_pct  = None
    vwap_anchored_dist_pct = None
    df_v: Optional[pd.DataFrame] = None   # carries vwap_session column when computed

    if volume_ok:
        df_v = add_session_vwap(window_piv)
        last_sv = float(df_v["vwap_session"].iloc[-1])
        if math.isfinite(last_sv) and close_i > 0:
            vwap_session_dist_pct = round(abs(close_i - last_sv) / close_i * 100, 4)

        anchor_idx = get_last_major_pivot_index(
            df_v, kind="low", lookback_pivots=5, prominence_horizon=20
        )
        if anchor_idx is None:
            anchor_idx = get_last_major_pivot_index(
                df_v, kind="high", lookback_pivots=5, prominence_horizon=20
            )
        if anchor_idx is not None:
            df_v = add_anchored_vwap(df_v, anchor_idx=anchor_idx)
            last_av = float(df_v["vwap_anchored"].iloc[-1])
            if math.isfinite(last_av) and close_i > 0:
                vwap_anchored_dist_pct = round(abs(close_i - last_av) / close_i * 100, 4)

    # ── Step 10: nearby-zone selection (caps O(n×zones)) ─────────────────
    nearby = _select_nearby(ltf_zones, close_i, params.max_nearby_zones)

    # ── Step 11: event detection ──────────────────────────────────────────
    event: Optional[Dict] = None

    # Gap fades run first — a gap-open reversal is more specific than a
    # generic bounce/sweep and should not be overridden by zone signals.
    gap = detect_gap(window_piv, atr)
    if gap is not None and gap["type"] in ("gap_fade_long", "gap_fade_short"):
        event = gap

    if event is None:
        event = detect_ath_breakout(window_piv, zone_tol)

    if event is None:
        sweep = detect_liquidity_sweeps(
            window_piv, nearby, zone_tol, lookback_bars=params.sweep_lookback
        )
        if sweep is not None:
            event = sweep
        else:
            struct = detect_breakout_or_bounce(window_piv, nearby, zone_tol)
            if struct is not None:
                event = struct

    # Extended detectors — run only when no primary event found yet.
    if event is None:
        event = detect_fvg(window_piv, zone_tol, atr)
    if event is None and df_v is not None:
        event = detect_vwap_reclaim(df_v, atr)   # df_v has vwap_session column
    if event is None:
        event = detect_bos(window_piv, zone_tol)
    if event is None:
        event = detect_orb(window_piv, atr, zone_tol)
    if event is None:
        event = detect_outside_bar(window_piv)
    if event is None:
        event = detect_inside_bar(window_piv)
    # Gap continuations: lowest priority (similar confidence to breakouts).
    if event is None and gap is not None:
        event = gap

    if event is None:
        return None

    # ── Step 12: gate checks ──────────────────────────────────────────────
    direction    = _DIRECTION.get(event["type"])
    ema_confirms = bool(
        (direction == "long"  and trend_bull) or
        (direction == "short" and trend_bear)
    )

    # RSI divergence confluence gate.
    rsi_div = detect_rsi_divergence(window_piv)
    rsi_divergence_aligned = bool(
        (rsi_div == "bullish" and direction == "long") or
        (rsi_div == "bearish" and direction == "short")
    )

    ltf_trend_aligned  = ema_confirms
    htf_trend_aligned: Optional[bool] = None
    mtf_aligned = True

    if htf_zones:
        near_htf    = _near_any_htf_zone(float(event["zone_center"]), alignment, zone_tol)
        htf_aligned = (
            (direction == "long"  and htf_bias == "bullish") or
            (direction == "short" and htf_bias == "bearish")
        )
        htf_trend_aligned = htf_aligned
        _fake = {"type": event["type"], "zone_center": event["zone_center"]}
        mtf_aligned = filter_signal_by_htf(_fake, htf_bias, alignment, zone_tol)
    else:
        near_htf = False

    # Signal status — mirrors live pipeline exactly.
    if not volume_ok:
        signal_status = "watchlist"
    elif vol_spike and ema_confirms and mtf_aligned:
        signal_status = "active"
    else:
        signal_status = "watchlist"

    # ── Step 13: confluence scoring ───────────────────────────────────────
    triggering_zone = _find_zone_by_center(nearby, event["zone_center"])
    conf = score_setup(
        event                  = event,
        zone                   = triggering_zone,
        atr14                  = atr,
        vol_ratio              = vol_ratio,
        ltf_trend_aligned      = ltf_trend_aligned,
        htf_trend_aligned      = htf_trend_aligned,
        vwap_session_dist_pct  = vwap_session_dist_pct,
        vwap_anchored_dist_pct = vwap_anchored_dist_pct,
        htf_bias_aligned       = (htf_trend_aligned is True),
        near_htf_zone          = near_htf,
        ref_ts                 = bar_ts,
        rsi_divergence_aligned = rsi_divergence_aligned,
    )

    # ── Step 14: position sizing ──────────────────────────────────────────
    if direction == "long":
        stop_price   = round(close_i - params.atr_stop_mult   * atr, 4)
        target_price = round(close_i + params.atr_target_mult * atr, 4)
    else:
        stop_price   = round(close_i + params.atr_stop_mult   * atr, 4)
        target_price = round(close_i - params.atr_target_mult * atr, 4)

    return {
        # Identity / metadata
        "ts":                    bar_ts,
        "event_type":            event["type"],
        "direction":             direction,
        # Zone
        "zone_center":           event["zone_center"],
        "zone_touches":          event["touches"],
        "zones_ltf_count":       len(ltf_zones),
        "zones_htf_count":       len(htf_zones),
        # Price / sizing
        "close":                 close_i,
        "entry_price":           close_i,
        "stop_price":            stop_price,
        "target_price":          target_price,
        "r_multiple":            params.r_multiple,
        "atr_14":                atr,
        "distance_pct":          round(abs(close_i - event["zone_center"]) / close_i * 100, 4),
        # Score
        "confluence_score":      conf["score"],
        "confluence_reasons":    conf["reasons"],
        # Status & confirmations
        "signal_status":         signal_status,
        "vol_spike":             vol_spike,
        "vol_ratio":             vol_ratio,
        "ema_confirms":          ema_confirms,
        "trend":                 trend,
        # HTF / MTF
        "htf_bias":              htf_bias,
        "mtf_aligned":           mtf_aligned,
        "near_htf_zone":         near_htf,
        # VWAP
        "vwap_session_dist_pct":  vwap_session_dist_pct,
        "vwap_anchored_dist_pct": vwap_anchored_dist_pct,
        # Outcome — populated by _resolve_outcomes
        "outcome":               None,
        "exit_bar":              None,
        "exit_ts":               None,
        "exit_price":            None,
        "return_pct":            None,
        "r_realized":            None,
        "bars_to_exit":          None,
    }


# ---------------------------------------------------------------------------
# Outcome resolution (look-forward from each triggered bar)
# ---------------------------------------------------------------------------


def _resolve_outcomes(
    setups: List[Dict[str, Any]],
    full_df: pd.DataFrame,
    params: BacktestParams,
) -> None:
    """
    Fill outcome fields in-place by scanning forward from each setup bar.

    For LONG:  loss when low ≤ stop_price; win when high ≥ target_price.
    For SHORT: loss when high ≥ stop_price; win when low  ≤ target_price.

    When both stop and target could trigger on the same bar (e.g. a wide
    gap), stop is checked first (conservative / realistic assumption).
    Setups that reach horizon without resolution are marked "timeout".
    """
    if not setups:
        return

    n      = len(full_df)
    highs  = full_df["high"].to_numpy(dtype=np.float64)
    lows   = full_df["low"].to_numpy(dtype=np.float64)
    opens  = full_df["open"].to_numpy(dtype=np.float64)
    ts_arr = full_df["ts"].to_numpy()

    for s in setups:
        i         = s["bar_index"]
        direction = s["direction"]
        atr       = s["atr_14"]

        # ── Execution-mode: determine actual entry price ───────────────────
        if params.entry_mode == "enter_on_next_open":
            if i + 1 >= n:
                # Signal fired on the last bar — no next open available.
                s["outcome"] = "no_fill"
                continue
            raw_open = opens[i + 1]
            slip = params.slippage_bps / 10_000
            # Slippage increases cost: long pays more, short receives less.
            entry = raw_open * (1.0 + slip) if direction == "long" else raw_open * (1.0 - slip)
            entry = round(entry, 4)
            # Recalculate stop/target maintaining same ATR distances from the
            # adjusted entry (keep risk/reward ratio identical to live pipeline).
            if direction == "long":
                stop   = round(entry - params.atr_stop_mult   * atr, 4)
                target = round(entry + params.atr_target_mult * atr, 4)
            else:
                stop   = round(entry + params.atr_stop_mult   * atr, 4)
                target = round(entry - params.atr_target_mult * atr, 4)
            s["entry_price"]  = entry
            s["stop_price"]   = stop
            s["target_price"] = target
        else:
            # enter_on_close: entry at the bar-i close (already stored).
            entry  = s["entry_price"]
            stop   = s["stop_price"]
            target = s["target_price"]

        outcome    = "timeout"
        exit_bar   = None
        exit_price = None

        end = min(i + 1 + params.outcome_horizon_bars, n)
        for j in range(i + 1, end):
            hi = highs[j]
            lo = lows[j]

            if direction == "long":
                if lo <= stop:          # stop hit first (conservative)
                    outcome = "loss"; exit_bar = j; exit_price = stop; break
                if hi >= target:
                    outcome = "win";  exit_bar = j; exit_price = target; break
            else:                       # short
                if hi >= stop:
                    outcome = "loss"; exit_bar = j; exit_price = stop; break
                if lo <= target:
                    outcome = "win";  exit_bar = j; exit_price = target; break

        s["outcome"] = outcome
        if exit_bar is not None:
            risk = abs(entry - stop)
            s["exit_bar"]    = exit_bar
            s["exit_ts"]     = ts_arr[exit_bar]
            s["exit_price"]  = exit_price
            s["bars_to_exit"] = exit_bar - i
            if direction == "long":
                s["return_pct"] = round((exit_price - entry) / entry * 100, 4)
                s["r_realized"] = round((exit_price - entry) / risk, 3) if risk > 0 else None
            else:
                s["return_pct"] = round((entry - exit_price) / entry * 100, 4)
                s["r_realized"] = round((entry - exit_price) / risk, 3) if risk > 0 else None


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def _compute_summary(
    setups: List[Dict[str, Any]],
    total_bars: int,
    params: BacktestParams,
) -> Dict[str, Any]:
    """Aggregate backtest statistics over all triggered setups."""
    if not setups:
        return {
            "total_bars":               total_bars,
            "total_setups":             0,
            "active_setups":            0,
            "watchlist_setups":         0,
            "closed":                   0,
            "wins":                     0,
            "losses":                   0,
            "timeouts":                 0,
            "win_rate":                 None,
            "avg_r":                    None,
            "avg_r_wins":               None,
            "avg_r_losses":             None,
            "max_consecutive_losses":   0,
            "max_drawdown_r":           0.0,
            "signal_frequency_per_100": 0.0,
            "false_breakout_rate":      None,
            "avg_bars_to_exit":         None,
            "by_event_type":            {},
            "params":                   _params_dict(params),
        }

    active   = [s for s in setups if s["signal_status"] == "active"]
    closed   = [s for s in setups if s["outcome"] in ("win", "loss")]
    wins     = [s for s in closed if s["outcome"] == "win"]
    losses   = [s for s in closed if s["outcome"] == "loss"]
    timeouts = [s for s in setups if s["outcome"] == "timeout"]

    def _mean(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else None

    win_rate     = len(wins)   / len(closed) if closed else None
    avg_r        = _mean([s.get("r_realized") for s in closed])
    avg_r_wins   = _mean([s.get("r_realized") for s in wins])
    avg_r_losses = _mean([s.get("r_realized") for s in losses])
    avg_bars     = _mean([s.get("bars_to_exit") for s in closed])

    # Max consecutive losses (in chronological order).
    max_consec = cur_streak = 0
    for s in closed:
        if s["outcome"] == "loss":
            cur_streak += 1
            max_consec  = max(max_consec, cur_streak)
        else:
            cur_streak = 0

    # Drawdown proxy: max peak-to-trough cumulative R.
    cum_r = peak_r = max_dd = 0.0
    for s in closed:
        cum_r += s.get("r_realized") or 0.0
        if cum_r > peak_r:
            peak_r = cum_r
        dd = peak_r - cum_r
        if dd > max_dd:
            max_dd = dd

    # False breakout rate: breakout/breakdown that resolved as losses.
    breakouts  = [s for s in setups if s["event_type"] in ("breakout_up", "breakdown_down")]
    bo_losses  = [s for s in breakouts if s["outcome"] == "loss"]
    fb_rate    = len(bo_losses) / len(breakouts) if breakouts else None

    # Per event-type breakdown.
    by_event: Dict[str, Any] = {}
    for s in setups:
        et = s["event_type"]
        if et not in by_event:
            by_event[et] = {"count": 0, "wins": 0, "losses": 0, "timeouts": 0}
        by_event[et]["count"] += 1
        oc = s["outcome"]
        if oc == "win":
            by_event[et]["wins"] += 1
        elif oc == "loss":
            by_event[et]["losses"] += 1
        else:
            by_event[et]["timeouts"] += 1

    def _r(v):
        return round(v, 3) if v is not None else None

    return {
        "total_bars":               total_bars,
        "total_setups":             len(setups),
        "active_setups":            len(active),
        "watchlist_setups":         len(setups) - len(active),
        "closed":                   len(closed),
        "wins":                     len(wins),
        "losses":                   len(losses),
        "timeouts":                 len(timeouts),
        "win_rate":                 round(win_rate, 4)  if win_rate is not None else None,
        "avg_r":                    _r(avg_r),
        "avg_r_wins":               _r(avg_r_wins),
        "avg_r_losses":             _r(avg_r_losses),
        "max_consecutive_losses":   max_consec,
        "max_drawdown_r":           round(max_dd, 3),
        "signal_frequency_per_100": round(len(setups) / total_bars * 100, 2),
        "false_breakout_rate":      round(fb_rate, 4) if fb_rate is not None else None,
        "avg_bars_to_exit":         round(avg_bars, 1) if avg_bars is not None else None,
        "by_event_type":            by_event,
        "params":                   _params_dict(params),
    }


def _params_dict(params: BacktestParams) -> Dict[str, Any]:
    return {
        "pivot_order":          params.pivot_order,
        "zone_atr_mult":        params.zone_atr_mult,
        "min_zone_touches":     params.min_zone_touches,
        "max_nearby_zones":     params.max_nearby_zones,
        "sweep_lookback":       params.sweep_lookback,
        "vol_spike_mult":       params.vol_spike_mult,
        "atr_stop_mult":        params.atr_stop_mult,
        "atr_target_mult":      params.atr_target_mult,
        "outcome_horizon_bars": params.outcome_horizon_bars,
        "min_confluence_score": params.min_confluence_score,
        "only_active":          params.only_active,
        "disable_htf":          params.disable_htf,
        "entry_mode":           params.entry_mode,
        "slippage_bps":         params.slippage_bps,
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def run_walk_forward(
    candles: List[Dict[str, Any]],
    params: Optional[BacktestParams] = None,
    train_bars: int = 500,
    test_bars: int = 100,
    step_bars: int = 100,
    htf_candles: Optional[List[Dict[str, Any]]] = None,
    symbol: str = "",
    timeframe: str = "",
) -> Dict[str, Any]:
    """
    Rolling walk-forward validation across non-overlapping test windows.

    For each fold:
      - Warm-up + train window : candles[fold_start : fold_start + train_bars]
        (used only to warm up the pipeline — indicators have enough history)
      - Test window            : candles[fold_start + train_bars :
                                         fold_start + train_bars + test_bars]
        (setups triggered here are the only ones counted in this fold's metrics)
      - Slide by step_bars and repeat.

    The "train" window is currently used for warm-up only, not parameter
    optimisation.  Grid search can be layered on top by calling this function
    with different BacktestParams configurations and comparing averaged metrics.

    Parameters
    ----------
    candles : list of dicts
        Full OHLCV history, ascending by ts.
    params : BacktestParams or None
    train_bars : int
        Bars used for warm-up / context before each test window.
        Must be >= params.min_bars so indicators are fully warmed up.
    test_bars : int
        Out-of-sample bars evaluated in each fold.
    step_bars : int
        How many bars to slide the window between folds.
        Use step_bars == test_bars for non-overlapping test periods.
    htf_candles : list of dicts or None
        HTF candles — filtered per fold to ts <= last fold candle.
    symbol, timeframe : str

    Returns
    -------
    dict with keys:
      "averaged"  — mean of key metrics across all folds
      "folds"     — list of per-fold summaries (no raw setups to keep size small)
      "all_setups"— combined setups from all test windows (for detailed analysis)
      "symbol", "timeframe", "params"
    """
    if params is None:
        params = BacktestParams()

    candles_sorted = sorted(candles, key=lambda x: x["ts"])
    n = len(candles_sorted)

    min_needed = train_bars + test_bars
    if n < min_needed:
        return {
            "symbol":    symbol,
            "timeframe": timeframe,
            "error":     f"Need >= {min_needed} candles for walk-forward, got {n}",
            "averaged":  {},
            "folds":     [],
            "all_setups": [],
        }

    folds:       List[Dict[str, Any]] = []
    all_setups:  List[Dict[str, Any]] = []
    fold_idx = 0
    start    = 0

    while start + min_needed <= n:
        test_start = start + train_bars
        test_end   = test_start + test_bars

        fold_candles = candles_sorted[start:test_end]

        # Slice HTF candles up to the last timestamp of this fold window.
        fold_htf: Optional[List[Dict]] = None
        if htf_candles:
            last_ts = pd.Timestamp(fold_candles[-1]["ts"])
            fold_htf = [c for c in htf_candles if pd.Timestamp(c["ts"]) <= last_ts]

        fold_all_setups, _ = run_backtest(
            candles     = fold_candles,
            params      = params,
            htf_candles = fold_htf,
            symbol      = symbol,
            timeframe   = timeframe,
        )

        # Keep only setups that triggered within the test window.
        # bar_index is relative to fold_candles (which starts at `start`),
        # so test-window bar indices lie in [train_bars, train_bars + test_bars).
        test_setups = [
            s for s in fold_all_setups
            if train_bars <= s["bar_index"] < train_bars + test_bars
        ]

        # Tag each setup with its fold for post-analysis.
        for s in test_setups:
            s["wf_fold"]       = fold_idx
            s["wf_train_start"] = start
            s["wf_test_start"]  = test_start
        all_setups.extend(test_setups)

        fold_summary = _compute_summary(test_setups, test_bars, params)
        fold_summary["fold"]       = fold_idx
        fold_summary["train_start"] = start
        fold_summary["train_end"]   = test_start
        fold_summary["test_end"]    = test_end

        folds.append(fold_summary)
        fold_idx += 1
        start    += step_bars

    # ── Average key metrics across folds ──────────────────────────────────
    avg_keys = [
        "win_rate", "avg_r", "avg_r_wins", "avg_r_losses",
        "max_drawdown_r", "false_breakout_rate", "signal_frequency_per_100",
        "avg_bars_to_exit",
    ]

    def _avg(key: str) -> Optional[float]:
        vals = [f[key] for f in folds if f.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    averaged: Dict[str, Any] = {k: _avg(k) for k in avg_keys}
    averaged.update({
        "total_folds":            len(folds),
        "train_bars":             train_bars,
        "test_bars":              test_bars,
        "step_bars":              step_bars,
        "total_setups_all_folds": len(all_setups),
        "total_wins_all_folds":   sum(f.get("wins",    0) for f in folds),
        "total_losses_all_folds": sum(f.get("losses",  0) for f in folds),
    })

    logger.info(
        "walk-forward | %s %s | folds=%d | setups=%d | avg_win_rate=%s | avg_r=%s",
        symbol or "?", timeframe or "?",
        len(folds), len(all_setups),
        averaged.get("win_rate"), averaged.get("avg_r"),
    )

    return {
        "symbol":    symbol,
        "timeframe": timeframe,
        "averaged":  averaged,
        "folds":     folds,
        "all_setups": all_setups,
        "params":    _params_dict(params),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the structure-first signal pipeline on historical candles"
    )

    g = p.add_argument_group("data")
    g.add_argument("--symbol",    required=True,              help="Ticker, e.g. AAPL")
    g.add_argument("--timeframe", default="1h",               help="Bar timeframe (default: 1h)")
    g.add_argument("--source",    default="yahoo",            help="DB source filter (default: yahoo)")
    g.add_argument("--lookback",  type=int, default=0,
                   help="Max bars to load from DB (0 = all available)")
    g.add_argument("--out",       default="backtest_results",
                   help="Output path stem without extension (default: backtest_results)")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])

    g2 = p.add_argument_group("strategy params")
    g2.add_argument("--pivot-order",     type=int,   default=3,
                    help="Swing half-window in bars (default 3)")
    g2.add_argument("--zone-atr-mult",   type=float, default=0.5,
                    help="Zone tolerance = ATR × this (default 0.5)")
    g2.add_argument("--min-touches",     type=int,   default=2,
                    help="Minimum pivot touches per zone (default 2)")
    g2.add_argument("--vol-spike-mult",  type=float, default=1.8,
                    help="Volume spike threshold multiplier (default 1.8)")
    g2.add_argument("--atr-stop-mult",   type=float, default=1.5,
                    help="Stop = close ± ATR × this (default 1.5)")
    g2.add_argument("--atr-target-mult", type=float, default=3.0,
                    help="Target = close ± ATR × this (default 3.0)")
    g2.add_argument("--horizon",         type=int,   default=20,
                    help="Bars forward for outcome resolution (default 20)")
    g2.add_argument("--min-score",       type=int,   default=0,
                    help="Skip setups below this confluence score (default 0)")
    g2.add_argument("--only-active",     action="store_true",
                    help="Skip watchlist setups (all gates must pass)")
    g2.add_argument("--event-types",     default=None,
                    help="Comma-separated allow-list of event types, e.g. "
                         "sweep_up,sweep_down,bounce_up,reject_down "
                         "(default: all event types)")
    g2.add_argument("--zone-method",     default="pivots",
                    choices=["pivots", "supply_demand"],
                    help="Zone detection method: pivots (default) or supply_demand")
    g2.add_argument("--regime-filter",   action="store_true",
                    help="Gate signal direction by regime symbol EMA50 "
                         "(skip counter-trend signals)")
    g2.add_argument("--regime-symbol",   default="SPY",
                    help="Ticker used as regime filter (default: SPY)")
    g2.add_argument("--no-htf",          action="store_true",
                    help="Disable HTF enrichment")
    g2.add_argument("--entry-mode",      default="enter_on_close",
                    choices=["enter_on_close", "enter_on_next_open"],
                    help="Entry timing: close of signal bar or open of next bar "
                         "(default: enter_on_close)")
    g2.add_argument("--slippage-bps",    type=float, default=0.0,
                    help="Slippage in basis points applied to next-open entry "
                         "(default: 0.0)")

    g3 = p.add_argument_group("walk-forward")
    g3.add_argument("--wf-train",  type=int, default=0,
                    help="Enable walk-forward: bars in each train/warm-up window "
                         "(0 = disabled, run single backtest)")
    g3.add_argument("--wf-test",   type=int, default=100,
                    help="Bars in each test window (default: 100)")
    g3.add_argument("--wf-step",   type=int, default=100,
                    help="Slide step between folds (default: 100 = non-overlapping)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    params = BacktestParams(
        pivot_order          = args.pivot_order,
        zone_atr_mult        = args.zone_atr_mult,
        min_zone_touches     = args.min_touches,
        vol_spike_mult       = args.vol_spike_mult,
        atr_stop_mult        = args.atr_stop_mult,
        atr_target_mult      = args.atr_target_mult,
        outcome_horizon_bars = args.horizon,
        min_confluence_score = args.min_score,
        only_active          = args.only_active,
        event_types          = [e.strip() for e in args.event_types.split(",")] if args.event_types else None,
        zone_method          = args.zone_method,
        regime_filter        = args.regime_filter,
        regime_symbol        = args.regime_symbol,
        disable_htf          = args.no_htf,
        entry_mode           = args.entry_mode,
        slippage_bps         = args.slippage_bps,
    )

    # Fetch candles from DB — reuses the same helper as the live agent runner.
    from backend.db.session import SessionLocal
    from backend.agent.runner import _fetch_candles

    db = SessionLocal()
    try:
        candles = _fetch_candles(
            db        = db,
            symbol    = args.symbol.upper(),
            timeframe = args.timeframe,
            source    = args.source,
            lookback  = args.lookback or 999_999,
        )
        regime_candles = None
        if args.regime_filter and args.regime_symbol.upper() != args.symbol.upper():
            regime_candles = _fetch_candles(
                db        = db,
                symbol    = args.regime_symbol.upper(),
                timeframe = args.timeframe,
                source    = args.source,
                lookback  = args.lookback or 999_999,
            )
    finally:
        db.close()

    if not candles:
        print(
            f"[backtest] No candles found for {args.symbol} {args.timeframe} "
            f"source={args.source}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Walk-forward or single backtest ───────────────────────────────────
    if args.wf_train > 0:
        wf = run_walk_forward(
            candles     = candles,
            params      = params,
            train_bars  = args.wf_train,
            test_bars   = args.wf_test,
            step_bars   = args.wf_step,
            symbol      = args.symbol.upper(),
            timeframe   = args.timeframe,
        )
        if "error" in wf:
            print(f"[walk-forward] error: {wf['error']}", file=sys.stderr)
            sys.exit(1)

        wf_json = Path(args.out).parent / (Path(args.out).stem + "_walkforward.json")
        wf_json.parent.mkdir(parents=True, exist_ok=True)
        with wf_json.open("w", encoding="utf-8") as fh:
            json.dump(wf, fh, indent=2, default=str)

        avg = wf["averaged"]
        print(f"[walk-forward] symbol:           {wf['symbol']}")
        print(f"[walk-forward] timeframe:        {wf['timeframe']}")
        print(f"[walk-forward] folds:            {avg['total_folds']}")
        print(f"[walk-forward] train/test/step:  {args.wf_train}/{args.wf_test}/{args.wf_step}")
        print(f"[walk-forward] total_setups:     {avg['total_setups_all_folds']}")
        print(f"[walk-forward] avg_win_rate:     {avg.get('win_rate')}")
        print(f"[walk-forward] avg_r:            {avg.get('avg_r')}")
        print(f"[walk-forward] avg_r_wins:       {avg.get('avg_r_wins')}")
        print(f"[walk-forward] avg_r_losses:     {avg.get('avg_r_losses')}")
        print(f"[walk-forward] avg_max_drawdown: {avg.get('max_drawdown_r')}")
        print(f"[walk-forward] avg_fb_rate:      {avg.get('false_breakout_rate')}")
        print(f"[walk-forward] avg_freq_per_100: {avg.get('signal_frequency_per_100')}")
        print(f"[walk-forward] saved → {wf_json}")
        if wf["folds"]:
            print("[walk-forward] per-fold win_rate:")
            for f in wf["folds"]:
                print(
                    f"               fold {f['fold']:>3}  "
                    f"test=[{f['train_end']}, {f['test_end']})  "
                    f"setups={f.get('total_setups', 0)}  "
                    f"win_rate={f.get('win_rate')}"
                )
    else:
        setups, summary = run_backtest(
            candles        = candles,
            params         = params,
            regime_candles = regime_candles,
            symbol         = args.symbol.upper(),
            timeframe      = args.timeframe,
        )

        save_results(setups, summary, args.out)

        print(f"[backtest] symbol:                 {summary.get('symbol')}")
        print(f"[backtest] timeframe:              {summary.get('timeframe')}")
        print(f"[backtest] entry_mode:             {params.entry_mode}")
        print(f"[backtest] slippage_bps:           {params.slippage_bps}")
        print(f"[backtest] total_bars:             {summary.get('total_bars')}")
        print(f"[backtest] total_setups:           {summary.get('total_setups')}")
        print(f"[backtest] active_setups:          {summary.get('active_setups')}")
        wins     = summary.get('wins', 0)
        losses   = summary.get('losses', 0)
        timeouts = summary.get('timeouts', 0)
        print(f"[backtest] wins/losses/timeouts:   {wins}/{losses}/{timeouts}")
        print(f"[backtest] win_rate:               {summary.get('win_rate')}")
        print(f"[backtest] avg_r:                  {summary.get('avg_r')}")
        print(f"[backtest] avg_r_wins:             {summary.get('avg_r_wins')}")
        print(f"[backtest] avg_r_losses:           {summary.get('avg_r_losses')}")
        print(f"[backtest] max_consecutive_losses: {summary.get('max_consecutive_losses')}")
        print(f"[backtest] max_drawdown_r:         {summary.get('max_drawdown_r')}")
        print(f"[backtest] false_breakout_rate:    {summary.get('false_breakout_rate')}")
        print(f"[backtest] signal_freq_per_100:    {summary.get('signal_frequency_per_100')}")
        print(f"[backtest] scan_time_ms:           {summary.get('scan_time_ms')}")
        if summary.get("by_event_type"):
            print("[backtest] by event type:")
            for et, stats in summary["by_event_type"].items():
                print(f"           {et:<20} count={stats['count']}  "
                      f"wins={stats['wins']}  losses={stats['losses']}  "
                      f"timeouts={stats['timeouts']}")

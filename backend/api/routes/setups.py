import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from backend.db.models import Setup

router = APIRouter(prefix="/setups", tags=["setups"])


def _row_to_dict(r: Setup) -> Dict[str, Any]:
    """Serialize a Setup row.  Legacy fields are always present; new
    institutional-lite fields are appended so existing clients are unaffected."""
    details: Dict[str, Any] = {}
    if r.details:
        try:
            details = json.loads(r.details)
        except (ValueError, TypeError):
            pass

    return {
        # ── Legacy fields (unchanged) ──────────────────────────────────────
        "id":            r.id,
        "run_id":        r.run_id,
        "symbol":        r.symbol,
        "timeframe":     r.timeframe,
        "source":        r.source,
        "ts":            r.ts,
        "close":         float(r.close)        if r.close        is not None else None,
        "ema_20":        float(r.ema_20)       if r.ema_20       is not None else None,
        "ema_50":        float(r.ema_50)       if r.ema_50       is not None else None,
        "rsi_14":        float(r.rsi_14)       if r.rsi_14       is not None else None,
        "atr_14":        float(r.atr_14)       if r.atr_14       is not None else None,
        "score":         float(r.score)        if r.score        is not None else None,
        "distance_pct":  float(r.distance_pct) if r.distance_pct is not None else None,
        "current_state": r.current_state,
        "trigger_type":  r.trigger_type,
        "created_at":    r.created_at,
        # ── New institutional-lite scalar fields ───────────────────────────
        "setup_type":       r.setup_type,
        "confluence_score": float(r.confluence_score) if r.confluence_score is not None else None,
        "vol_spike":        r.vol_spike,
        "htf_aligned":      r.htf_aligned,
        "signal_status":    r.signal_status,
        # ── Rich details (parsed from JSON; None for pre-migration rows) ───
        "event_type":             details.get("event_type"),
        "zone_center":            details.get("zone_center"),
        "zone_touches":           details.get("zone_touches"),
        "zones_ltf_count":        details.get("zones_ltf_count"),
        "zones_htf_count":        details.get("zones_htf_count"),
        "htf_bias":               details.get("htf_bias"),
        "near_htf_zone":          details.get("near_htf_zone"),
        "mtf_aligned":            details.get("mtf_aligned"),
        "vwap_session":           details.get("vwap_session"),
        "vwap_anchored":          details.get("vwap_anchored"),
        "vwap_session_dist_pct":  details.get("vwap_session_dist_pct"),
        "vwap_anchored_dist_pct": details.get("vwap_anchored_dist_pct"),
        "zones_ltf":              details.get("zones_ltf"),
        "zones_htf":              details.get("zones_htf"),
        "confluence_reasons":     details.get("confluence_reasons"),
        "vol_ratio":              details.get("vol_ratio"),
        "ema_confirms":           details.get("ema_confirms"),
        "data_quality":           details.get("data_quality"),
    }


@router.get("/latest")
def get_latest_setups(
    timeframe: str = Query("1h"),
    source: Optional[str] = Query(None),
    signal_status: Optional[str] = Query(
        None, description="Filter by signal status: active or watchlist"
    ),
    limit: int = Query(50, ge=1, le=500),
):
    from backend.db.session import SessionLocal
    db: Session = SessionLocal()

    try:
        q = db.query(Setup).filter(Setup.timeframe == timeframe)
        if source:
            q = q.filter(Setup.source == source)
        if signal_status:
            q = q.filter(Setup.signal_status == signal_status)

        rows = q.order_by(Setup.ts.desc(), Setup.score.asc()).limit(limit).all()
        return [_row_to_dict(r) for r in rows]

    finally:
        db.close()

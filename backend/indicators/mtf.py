"""
Multi-timeframe (MTF) structure alignment.

Philosophy
----------
Higher-timeframe (HTF) zones define the primary structure.  Lower-timeframe
(LTF) signals are only acted on when they:

  1. Agree with the HTF directional bias (trend alignment), OR
  2. Occur as a sweep reversal at an HTF structural level (counter-trend
     but high-probability: smart money flushes retail stops then reverses).

Three public entry points
-------------------------
align_zones(htf_zones, ltf_zones, tol) -> dict
    Link each LTF zone to the nearest HTF zone within *tol*.  Returns a
    structured dict separating linked pairs, unlinked LTF zones, and
    HTF-only zones.  Zone kind (support/resistance) is NOT enforced during
    matching — proximity alone governs linkage so that price levels that
    flip between support and resistance are handled correctly.

get_htf_bias(htf_df) -> "bullish" | "bearish" | "neutral"
    Classify the HTF trend using EMA50 of close.  Requires at least a few
    dozen bars for the EMA to be meaningful; fewer bars may produce
    unreliable bias.

filter_signal_by_htf(signal, htf_bias, alignment, zone_tol) -> bool
    Gate a LTF signal against HTF bias.  Returns True if the signal
    should be acted upon, False if it should be suppressed.

Tolerance convention
--------------------
The caller is responsible for computing *tol*.  Recommended:

    tol = max(htf_atr14 * 0.5, ltf_atr14 * 0.5)

or a fixed percentage of price (e.g. 0.003 * last_close).  This mirrors
the ATR-based zone_tol used throughout the rest of the indicator stack.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from backend.indicators.trend import add_trend_filter
from backend.indicators.zones import Zone


# ---------------------------------------------------------------------------
# Direction sets — shared constants
# ---------------------------------------------------------------------------

# LTF event types that imply upward / bullish price action.
_BULLISH_EVENT_TYPES: frozenset = frozenset(
    {"breakout_up", "bounce_up", "sweep_down"}
)

# LTF event types that imply downward / bearish price action.
_BEARISH_EVENT_TYPES: frozenset = frozenset(
    {"breakdown_down", "reject_down", "sweep_up"}
)

# Sweep event types that receive the HTF-level override.
_SWEEP_TYPES: frozenset = frozenset({"sweep_up", "sweep_down"})


# ---------------------------------------------------------------------------
# Public API — zone alignment
# ---------------------------------------------------------------------------


def align_zones(
    htf_zones: List[Zone],
    ltf_zones: List[Zone],
    tol: float,
) -> Dict:
    """
    Link each LTF zone to its nearest HTF zone within *tol*.

    Uses vectorised pairwise distance (numpy broadcasting) so the inner loop
    runs in C even for large zone lists; typical zone counts (< 30 per TF)
    make this effectively O(1).

    Parameters
    ----------
    htf_zones : List[Zone]
        Zones built from the higher-timeframe candles.
    ltf_zones : List[Zone]
        Zones built from the lower-timeframe candles.
    tol : float
        Maximum center-to-center distance (in price units) for two zones to
        be considered aligned.  Recommended: ``max(htf_atr14, ltf_atr14) * 0.5``.

    Returns
    -------
    dict with keys:

        ``linked``
            List of ``{"ltf": Zone, "htf": Zone, "distance": float}``.
            Multiple LTF zones may link to the same HTF zone.
            Sorted ascending by *distance*.

        ``unlinked_ltf``
            LTF zones for which no HTF zone was within *tol*.
            These zones have no higher-timeframe confluence.

        ``htf_only``
            HTF zones not referenced by any linked pair.
            These are major structural levels with no LTF echo — still
            important as price magnets but not currently active on the LTF.

    Raises
    ------
    ValueError
        If *tol* is not positive.
    """
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol!r}")

    # --- Early-exit for degenerate inputs ---------------------------------
    if not htf_zones or not ltf_zones:
        return {
            "linked": [],
            "unlinked_ltf": list(ltf_zones),
            "htf_only": list(htf_zones),
        }

    # --- Vectorised pairwise distance -------------------------------------
    ltf_centers = np.array([z.center for z in ltf_zones], dtype=np.float64)  # (L,)
    htf_centers = np.array([z.center for z in htf_zones], dtype=np.float64)  # (H,)

    # Shape (L, H): absolute center-to-center distance for every pair.
    dist_matrix = np.abs(ltf_centers[:, np.newaxis] - htf_centers[np.newaxis, :])

    nearest_htf_idx = dist_matrix.argmin(axis=1)   # (L,) — closest HTF for each LTF
    nearest_dist    = dist_matrix.min(axis=1)       # (L,) — that closest distance

    within_tol = nearest_dist <= tol                # (L,) bool mask

    # --- Build output lists (O(L) + O(H) Python loops, L and H are small) -
    linked: List[Dict] = []
    unlinked_ltf: List[Zone] = []
    linked_htf_set: set = set()

    for ltf_i, (is_linked, htf_i, dist) in enumerate(
        zip(within_tol, nearest_htf_idx, nearest_dist)
    ):
        if is_linked:
            linked.append(
                {
                    "ltf":      ltf_zones[ltf_i],
                    "htf":      htf_zones[int(htf_i)],
                    "distance": float(dist),
                }
            )
            linked_htf_set.add(int(htf_i))
        else:
            unlinked_ltf.append(ltf_zones[ltf_i])

    # HTF zones that were not matched by any LTF zone.
    htf_only = [
        htf_zones[i]
        for i in range(len(htf_zones))
        if i not in linked_htf_set
    ]

    # Sort linked pairs by distance for caller convenience.
    linked.sort(key=lambda e: e["distance"])

    return {
        "linked":       linked,
        "unlinked_ltf": unlinked_ltf,
        "htf_only":     htf_only,
    }


# ---------------------------------------------------------------------------
# Public API — HTF bias
# ---------------------------------------------------------------------------


def get_htf_bias(
    htf_df: pd.DataFrame,
) -> Literal["bullish", "bearish", "neutral"]:
    """
    Classify the HTF directional bias using EMA50 of close.

    Parameters
    ----------
    htf_df : pd.DataFrame
        Higher-timeframe candle DataFrame.  Must contain ``close``.
        Fewer than ~50 bars may produce an unreliable EMA50; the result
        is still returned but should be treated with lower confidence.

    Returns
    -------
    "bullish"
        Last bar's close is above its EMA50.
    "bearish"
        Last bar's close is below its EMA50.
    "neutral"
        Close exactly equals EMA50 (rare in practice).
    """
    enriched = add_trend_filter(htf_df)
    last = enriched.iloc[-1]
    if last["trend_bull"]:
        return "bullish"
    if last["trend_bear"]:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API — signal filter
# ---------------------------------------------------------------------------


def filter_signal_by_htf(
    signal: Optional[Dict],
    htf_bias: Literal["bullish", "bearish", "neutral"],
    alignment: Dict,
    zone_tol: float,
) -> bool:
    """
    Decide whether a LTF signal should be acted on given the HTF context.

    Rules
    -----
    1. ``signal`` is None                → **False** (nothing to act on).
    2. ``htf_bias`` is ``"neutral"``     → **True**  (no bias to gate against).
    3. Sweep at an HTF structural level  → **True**  regardless of bias direction.
       This captures the *stop-hunt reversal* pattern: smart money sweeps an
       HTF level to flush retail stops, then reverses.  The sweep direction
       may be counter-trend; proximity to the HTF level is what matters.
    4. Signal direction agrees with bias → **True**.
    5. Otherwise                         → **False** (counter-trend, not at HTF).

    Directional mapping
    -------------------
    Bullish-aligned types : ``breakout_up``, ``bounce_up``, ``sweep_down``
    Bearish-aligned types : ``breakdown_down``, ``reject_down``, ``sweep_up``

    Parameters
    ----------
    signal : dict or None
        Output of ``detect_breakout_or_bounce`` or ``detect_liquidity_sweeps``.
        Must contain ``"type"`` and ``"zone_center"`` keys.
    htf_bias : str
        Output of ``get_htf_bias``.
    alignment : dict
        Output of ``align_zones``.  Used to locate HTF zone centers for the
        sweep-at-HTF-level override.
    zone_tol : float
        Maximum distance from a signal's zone center to an HTF zone center
        for the sweep override to apply.  Typically the same value passed
        to ``align_zones``.

    Returns
    -------
    bool
    """
    if signal is None:
        return False

    if htf_bias == "neutral":
        return True

    sig_type: str = signal["type"]
    sig_center: Optional[float] = signal.get("zone_center")

    # --- Rule 3: sweep at HTF level overrides bias check -----------------
    if sig_type in _SWEEP_TYPES and sig_center is not None:
        if _near_any_htf_zone(float(sig_center), alignment, zone_tol):
            return True

    # --- Rule 4: direction alignment -------------------------------------
    if htf_bias == "bullish":
        return sig_type in _BULLISH_EVENT_TYPES
    if htf_bias == "bearish":
        return sig_type in _BEARISH_EVENT_TYPES

    return True  # unreachable unless htf_bias is an unexpected value


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _near_any_htf_zone(
    zone_center: float,
    alignment: Dict,
    tol: float,
) -> bool:
    """
    Return True if *zone_center* is within *tol* of any HTF zone center
    (whether the HTF zone is linked or htf_only).
    """
    # Collect all unique HTF zone centers from the alignment output.
    htf_centers: List[float] = [e["htf"].center for e in alignment.get("linked", [])]
    htf_centers += [z.center for z in alignment.get("htf_only", [])]

    if not htf_centers:
        return False

    dists = np.abs(np.array(htf_centers, dtype=np.float64) - zone_center)
    return bool(dists.min() <= tol)

"""backend/notifications/dispatcher.py

Central notification dispatcher.

Receives a formatted alert dict, checks quality thresholds, then sends
an Expo push notification to registered mobile devices.

Usage
-----
    from backend.notifications.dispatcher import dispatch_signal
    dispatch_signal(alert_dict)   # fire-and-forget, never raises

The dispatcher is intentionally silent on failure — a broken notification
channel must never interrupt the trading pipeline.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from backend.config import settings
from backend.notifications.channels.push import send_push

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------


def _passes_threshold(alert: Dict[str, Any]) -> bool:
    """Return True when the alert clears all notification thresholds."""
    if settings.notify_active_only and alert.get("signal_status") != "active":
        return False

    probability = alert.get("probability") or 0.0
    if probability < settings.notify_min_probability:
        return False

    confluence = alert.get("confluence_score") or 0
    if confluence < settings.notify_min_confluence:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispatch_signal(alert: Dict[str, Any]) -> None:
    """
    Send an alert as an Expo push notification to all registered devices.

    Runs in a daemon thread so the pipeline is never blocked.
    Never raises — all exceptions are caught and logged.

    Parameters
    ----------
    alert : dict
        Formatted alert payload from backend.signals.explain.format_alert().
        Must contain at minimum: symbol, event_type, signal_status,
        probability, confluence_score.
    """
    symbol = alert.get("symbol", "?")
    prob   = alert.get("probability_pct", "?")

    if not _passes_threshold(alert):
        logger.debug(
            "Signal below notification threshold | %s | prob=%s | status=%s",
            symbol, prob, alert.get("signal_status"),
        )
        return

    tokens_raw = settings.expo_push_tokens.strip()
    if not tokens_raw:
        logger.debug("No Expo push tokens configured — skipping push for %s", symbol)
        return

    tokens = [t.strip() for t in tokens_raw.split(",") if t.strip()]

    logger.info(
        "Dispatching push notification | %s | %s | %s | prob=%s",
        symbol, alert.get("event_type"), alert.get("direction"), prob,
    )

    t = threading.Thread(
        target=_safe_push,
        args=(tokens, alert),
        daemon=True,
    )
    t.start()


def _safe_push(tokens: list, alert: Dict[str, Any]) -> None:
    try:
        send_push(tokens, alert)
    except Exception as e:
        logger.error("Push notification failed: %s", e)

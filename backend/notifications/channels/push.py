"""backend/notifications/channels/push.py

Expo push notification channel.

Sends push notifications to mobile devices registered via the Expo
push notification service.  Tokens are set via EXPO_PUSH_TOKENS in .env
(comma-separated list of ExponentPushToken[...] strings).

This is a stub until the Expo frontend is built.  Once users register
their device token through the mobile app, add it to EXPO_PUSH_TOKENS
and push alerts will start flowing immediately.

Expo push docs: https://docs.expo.dev/push-notifications/sending-notifications/
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
_TIMEOUT_S     = 10


def send_push(tokens: List[str], alert: Dict[str, Any]) -> bool:
    """
    Send a push notification to one or more Expo push tokens.

    Returns True if all messages were accepted, False on any failure.
    Does not raise — notification failures must never crash the pipeline.
    """
    if not tokens:
        return False

    symbol   = alert.get("symbol", "?")
    prob_pct = alert.get("probability_pct", "?")
    direction_label = alert.get("direction_label", alert.get("direction", ""))
    event_type = alert.get("event_type", "signal")
    explanation = alert.get("explanation", "")

    title = f"{symbol} — {prob_pct} {direction_label}"
    body  = explanation[:150] + "…" if len(explanation) > 150 else explanation

    messages = [
        {
            "to":    token,
            "title": title,
            "body":  body,
            "data":  {
                "symbol":      symbol,
                "event_type":  event_type,
                "probability": alert.get("probability"),
                "direction":   alert.get("direction"),
            },
            "sound": "default",
            "priority": "high",
        }
        for token in tokens
    ]

    payload = json.dumps(messages).encode("utf-8") if len(messages) > 1 else json.dumps(messages[0]).encode("utf-8")

    try:
        req = urllib.request.Request(
            _EXPO_PUSH_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept":       "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            result = json.loads(resp.read().decode())
            logger.info("Push sent | %s | tokens=%d | result=%s", symbol, len(tokens), result)
            return True
    except Exception as e:
        logger.warning("Push notification failed: %s", e)
        return False

"""backend/api/routes/notifications.py

Push token registration endpoint.

The mobile app calls POST /notifications/register on startup with its
Expo push token. The backend appends it to EXPO_PUSH_TOKENS in the
runtime settings so the dispatcher can reach the device immediately.

Note: tokens are stored in-memory only. To persist across restarts,
add the token to EXPO_PUSH_TOKENS in your .env file.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings

router = APIRouter(prefix="/notifications", tags=["notifications"])
logger = logging.getLogger(__name__)


class TokenRequest(BaseModel):
    token: str


@router.post("/register")
def register_token(body: TokenRequest):
    """Register an Expo push token for this device."""
    token = body.token.strip()
    if not token:
        return {"status": "ignored", "reason": "empty token"}

    existing = [t.strip() for t in settings.expo_push_tokens.split(",") if t.strip()]

    if token in existing:
        return {"status": "already_registered"}

    existing.append(token)
    settings.expo_push_tokens = ",".join(existing)

    logger.info("Push token registered: %s", token)
    return {"status": "registered", "token": token}

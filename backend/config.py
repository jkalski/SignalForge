from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./signalforge.db", alias="DATABASE_URL")
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")

    # ── Notifications ──────────────────────────────────────────────────────
    # Expo push token(s) from the mobile app, comma-separated (leave blank to disable)
    expo_push_tokens: str = Field(default="", alias="EXPO_PUSH_TOKENS")

    # Thresholds — only notify when signal clears these bars
    notify_min_probability:  float = Field(default=0.55, alias="NOTIFY_MIN_PROBABILITY")
    notify_min_confluence:   int   = Field(default=55,   alias="NOTIFY_MIN_CONFLUENCE")
    notify_active_only:      bool  = Field(default=True, alias="NOTIFY_ACTIVE_ONLY")

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")

settings = Settings()
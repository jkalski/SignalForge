from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = Field(default="sqlite:///./signalforge.db", alias="DATABASE_URL")
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")

settings = Settings()
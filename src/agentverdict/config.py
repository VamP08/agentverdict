"""Application settings, loaded from AGENTVERDICT_* environment variables and .env."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTVERDICT_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./agentverdict.db"
    debug: bool = False

    # Judging (M2). The API key also falls back to the conventional GROQ_API_KEY.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENTVERDICT_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"
    judge_model: str = "llama-3.3-70b-versatile"
    judge_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

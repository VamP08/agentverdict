"""Application settings, loaded from AGENTVERDICT_* environment variables and .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTVERDICT_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./agentverdict.db"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

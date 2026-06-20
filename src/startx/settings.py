"""Typed configuration. Secrets come from environment / .env (never hardcoded)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    cache_dir: Path = Path("data/cache")
    history_start: str = "2010-01-01"
    rate_limit_per_min: int = 2800

    def require_key(self) -> str:
        if not self.fmp_api_key:
            raise RuntimeError(
                "FMP_API_KEY is not set. Copy .env.example to .env and add your key "
                "(get/regenerate it in your FMP dashboard)."
            )
        return self.fmp_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

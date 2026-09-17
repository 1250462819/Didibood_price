"""Application settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "Didibood Price"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8093

    DATABASE_URL: str = Field(
        default="postgresql://crawl_user:changeme@127.0.0.1:5432/crawl",
    )

    MODEL_DIR: Path = Field(default=SERVICE_ROOT / "artifacts" / "models")
    DATA_CACHE_DIR: Path = Field(default=SERVICE_ROOT / "artifacts" / "datasets")

    MIN_TRAINING_ROWS: int = 80
    TRAIN_TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42
    OUTLIER_IQR_MULTIPLIER: float = 1.5

    COMPARABLE_AREA_TOLERANCE_PCT: float = 0.15
    COMPARABLE_MIN_SAMPLE: int = 5
    # Comparables use PLP rows last seen within this window (live market snapshot).
    COMPARABLE_RECENT_DAYS: int = 30

    # Didibood default rent conversion (Toman): 10M deposit ↔ 300K monthly
    RENT_RATIO_DEPOSIT_TOMAN: float = 10_000_000.0
    RENT_RATIO_MONTHLY_TOMAN: float = 300_000.0

    # Rent-only training caps (Toman) — sale pipeline ignores these
    RENT_MAX_DEPOSIT_TOMAN: float = 5_000_000_000.0
    RENT_MAX_MONTHLY_TOMAN: float = 200_000_000.0
    RENT_MAX_EQUIVALENT_TOMAN: float = 8_000_000_000.0
    RENT_MIN_EQUIVALENT_TOMAN: float = 10_000_000.0
    RENT_OUTLIER_IQR_MULTIPLIER: float = 1.0
    # Divar sometimes stores monthly rent in price_total when price_per_unit is empty.
    RENT_MISMAPPED_DEPOSIT_MAX_TOMAN: float = 100_000_000.0

    # Neighbourhood analytics reads a prebuilt parquet; older than this and a
    # refresh runs in the background while the stale copy is still served.
    NEIGHBOURHOOD_CACHE_HOURS: int = 12
    NEIGHBOURHOOD_MAX_MONTHS: int = 12

    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])


settings = Settings()
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

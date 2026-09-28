"""Настройки без сетевых ключей и без зависимости от рабочего каталога."""

from pathlib import Path

from pydantic import Field, field_validator

from app.backend.contracts import Contract
from app.identity import default_data_dir, validate_data_dir


class BackendSettings(Contract):
    data_dir: Path = Field(default_factory=default_data_dir)
    collection_workers: int = Field(default=2, ge=1, le=4, strict=True)
    page_size: int = Field(default=100, ge=1, le=1000, strict=True)
    timeout_seconds: float = Field(default=15.0, ge=1, le=60, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=5, strict=True)
    max_response_bytes: int = Field(default=5_000_000, ge=1024, le=20_000_000, strict=True)
    max_pending_jobs: int = Field(default=20, ge=1, le=100, strict=True)
    history_period_delay_seconds: float = Field(default=1.0, ge=0, le=30, allow_inf_nan=False)

    @field_validator("data_dir")
    @classmethod
    def absolute_dir(cls, value):
        return validate_data_dir(value)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "documents.sqlite3"

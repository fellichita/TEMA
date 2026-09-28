"""Small, serializable contracts for the new local MVP."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    topic: str = Field(min_length=2, max_length=500)
    start_year: int = Field(default=2020, ge=1900, le=9998)
    end_year: int = Field(default=2025, ge=1900, le=9998)
    top_k: int = Field(default=15, ge=1, le=15)
    max_documents: int = Field(default=20_000, ge=20, le=30_000)
    relevance_mode: Literal["lexical", "semantic"] = "lexical"

    @model_validator(mode="after")
    def valid_window(self):
        if self.end_year - self.start_year < 3 or self.end_year - self.start_year > 29:
            raise ValueError("Для анализа нужны 4–30 завершённых лет.")
        if self.end_year >= datetime.now(timezone.utc).year:
            raise ValueError("Последний год анализа должен быть завершён.")
        return self


class AnalysisInputError(ValueError):
    """Safe, user-facing input error."""

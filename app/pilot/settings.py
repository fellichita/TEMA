"""Strict non-secret preferences; credentials live only in CredentialStore."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pilot.llm import ProviderConfig
from app.runtime.jobs import TaskFailure

CollectionProfile = Literal["fast", "deep"]

# Два режима: быстрый предварительный обзор и глубокий анализ с полной выборкой,
# самой длинной историей и патентами. Время определяет в первую очередь число
# кандидатов, которых называет модель: замер «solid-state batteries» — 406 из 600
# секунд быстрого режима уходили на именование, пока пул был общим (60). Пока ТОП
# не заполнен, анализ перебирает весь пул, поэтому пул у режимов разный.
COLLECTION_PROFILES: dict[str, dict[str, int | bool]] = {
    "fast": {"discovery_documents": 1000, "candidate_limit": 8, "candidate_attempts": 24,
             "history_enabled": True, "patents_enabled": False},
    "deep": {"discovery_documents": 10000, "candidate_limit": 15, "candidate_attempts": 60,
             "history_enabled": True, "patents_enabled": True},
}
# Исторический бюджет режима: (документов на кандидата, новых документов всего).
PROFILE_HISTORY_LIMITS: dict[str, tuple[int, int]] = {"fast": (150, 1500), "deep": (3000, 20000)}


class PilotSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    # "local" runs the installed instruct model: no key, no network, no cost,
    # and the same three jobs an online provider performs. It is the default
    # because an analysis without any AI cannot confirm a technology at all, and
    # a key is the one thing a new profile does not have.
    provider: Literal["deepseek", "yandex", "local"] = "local"
    yandex_folder: str = Field(default="", max_length=80, pattern=r"^[a-zA-Z0-9_-]*$")
    model_name: str = "deepseek-v4-flash"
    model_version: str = "DeepSeek-V4-Flash-0731"
    pricing_version: str = "official-2026-09-10-peak"
    currency: Literal["USD", "RUB"] = "USD"
    input_per_million_micro: int = Field(default=440000, gt=0, le=10**12, strict=True)
    output_per_million_micro: int = Field(default=1320000, gt=0, le=10**12, strict=True)
    run_cost_micro: int = Field(default=500000, ge=0, le=1000000000, strict=True)
    day_cost_micro: int = Field(default=2000000, ge=0, le=10000000000, strict=True)
    discovery_documents: int = Field(default=10000, ge=100, le=10000, strict=True)
    candidate_limit: int = Field(default=15, ge=1, le=15, strict=True)
    # Сколько обнаруженных кандидатов анализ готов назвать и проверить, пока
    # заполняет ТОП. Сохранённые до появления поля настройки получают прежние 60.
    candidate_attempts: int = Field(default=60, ge=1, le=60, strict=True)
    history_enabled: bool = Field(default=True, strict=True)
    patents_enabled: bool = Field(default=False, strict=True)
    external_ai_allowed: bool = Field(default=True, strict=True)
    wordstat_api_enabled: bool = Field(default=False, strict=True)
    wordstat_folder_id: str = Field(default="", max_length=50, pattern=r"^[A-Za-z0-9_-]*$")
    wordstat_hourly_cap: int = Field(default=3, ge=1, le=80, strict=True)
    wordstat_daily_cap: int = Field(default=20, ge=1, le=200, strict=True)

    @model_validator(mode="after")
    def valid_settings(self) -> Self:
        if self.day_cost_micro < self.run_cost_micro:
            raise ValueError("Дневной лимит не может быть меньше лимита одного анализа.")
        if self.candidate_attempts < self.candidate_limit:
            raise ValueError("Кандидатов на проверку не может быть меньше размера ТОП.")
        if self.wordstat_api_enabled and len(self.wordstat_folder_id) < 6:
            raise ValueError("Для Wordstat API нужен идентификатор каталога Yandex Cloud.")
        if self.provider == "local":
            return self
        if self.provider == "deepseek":
            self.provider_config()
        elif self.yandex_folder:
            self.provider_config()
        elif self.currency != "RUB":
            raise ValueError("Для Yandex требуется тариф в рублях.")
        return self

    def provider_config(self) -> ProviderConfig:
        if self.provider == "local":
            raise TaskFailure("Локальная модель не использует тариф внешнего провайдера.")
        if self.provider == "yandex" and not self.yandex_folder:
            raise TaskFailure("В настройках Yandex укажите идентификатор каталога.")
        model = (f"gpt://{self.yandex_folder}/{self.model_name}" if self.provider == "yandex" else self.model_name)
        return ProviderConfig(provider=self.provider, model=model, model_version=self.model_version,
                              pricing_version=self.pricing_version, currency=self.currency,
                              input_per_million_micro=self.input_per_million_micro,
                              output_per_million_micro=self.output_per_million_micro)

    @classmethod
    def for_provider(cls, provider: str) -> PilotSettings:
        if provider == "deepseek":
            # Explicit: the class default is the local model, not this provider.
            return cls(provider="deepseek")
        if provider == "local":
            # Tariff fields keep their defaults and are never charged; the day
            # and run limits stay in place as a bound on requests, not money.
            return cls(provider="local")
        if provider == "yandex":
            return cls(provider="yandex", model_name="deepseek-v4-flash", model_version="yandex-deepseek-v4-flash",
                       currency="RUB", input_per_million_micro=300000000, output_per_million_micro=500000000,
                       run_cost_micro=100000000, day_cost_micro=300000000)
        raise ValueError("Поставщик AI не поддерживается.")


def apply_collection_profile(settings: PilotSettings, profile: CollectionProfile) -> PilotSettings:
    """Apply an explicit per-run collection profile without changing model identity or saved preferences."""
    return PilotSettings.model_validate(settings.model_dump(mode="python") | COLLECTION_PROFILES[profile])


def load_settings(data_dir: Path) -> PilotSettings:
    from app.pilot.reports import open_local_regular

    path = data_dir / "settings.json"
    if not path.exists():
        return PilotSettings()
    try:
        with open_local_regular(path) as handle:
            data = handle.read(32001)
        if len(data) > 32000:
            raise ValueError("Oversized settings")
        return PilotSettings.model_validate_json(data)
    except (ValueError, OSError):
        raise TaskFailure("Настройки повреждены. Откройте настройки и сохраните проверенные значения заново.") from None


def save_settings(data_dir: Path, settings: PilotSettings) -> None:
    validated = PilotSettings.model_validate_json(settings.model_dump_json())
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "settings.json"
    temporary = path.with_suffix("." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(validated.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def public_settings(settings: PilotSettings) -> dict:
    values = settings.model_dump(mode="json")
    if settings.provider == "local":
        return values
    if settings.provider != "yandex" or settings.yandex_folder:
        values["tariff"] = asdict(settings.provider_config())
    return values

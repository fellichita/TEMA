"""Bounded human-approved query profiles and provider-specific request plans."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from app.pilot.contracts import content_hash
from app.pilot.multisource.contracts import QueryProfile, QueryTerm, SignalPolicy
from app.pilot.multisource.store import object_digest
from app.pilot.query import normalize_query
from app.runtime.jobs import TaskFailure

QUERY_COMPILER_VERSION = "wordstat-searchapi-monthly-plain/1"
_OPERATOR_CHARS = frozenset('!+-"[]()|')


def normalize_term(value: str) -> str:
    """NFKC and whitespace only: C++, 3D and word order retain their meaning."""
    if not isinstance(value, str):
        raise TaskFailure("Поисковая фраза должна быть текстом.")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(normalized) <= 400 or not any(character.isalpha() for character in normalized):
        raise TaskFailure("Поисковая фраза должна содержать название технологии.")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in normalized):
        raise TaskFailure("Поисковая фраза содержит управляющие символы.")
    return normalized


def build_manual_profile(query: str, definition: str, *, seed_terms: tuple[str, ...],
                         primary_phrase: str | None = None, exclusions: tuple[str, ...] = (),
                         confirmed_at: datetime | None = None, previous: QueryProfile | None = None,
                         previous_hash: str | None = None) -> QueryProfile:
    """Make a new immutable vocabulary version without requiring paid AI translation."""
    query = normalize_query(query)
    definition = " ".join(unicodedata.normalize("NFKC", definition).split()) if isinstance(definition, str) else ""
    if not 1 <= len(definition) <= 3000 or not any(char.isalpha() for char in definition):
        raise TaskFailure("Определение технологической области пустое или слишком длинное.")
    if len(seed_terms) > 50 or len(exclusions) > 40:
        raise TaskFailure("Слишком много поисковых терминов или исключений.")
    if (previous is None) != (previous_hash is None):
        raise TaskFailure("Для новой версии требуется предыдущий профиль и его идентификатор.")
    if previous is not None and object_digest(previous) != previous_hash:
        raise TaskFailure("Идентификатор предыдущего профиля не совпадает с его содержимым.")
    if previous is not None and previous.original_query != query:
        raise TaskFailure("Изменение области требует нового профиля.")
    date_confirmed = confirmed_at or datetime.now(timezone.utc)
    if date_confirmed.tzinfo is None or date_confirmed.utcoffset() is None:
        raise TaskFailure("Дата подтверждения должна включать часовой пояс.")
    clean_terms = tuple(normalize_term(item) for item in seed_terms)
    if len({item.casefold() for item in clean_terms}) != len(clean_terms):
        raise TaskFailure("Повторяющиеся названия технологии.")
    if primary_phrase is not None:
        primary_phrase = normalize_term(primary_phrase)
        if primary_phrase.casefold() not in {item.casefold() for item in clean_terms}:
            raise TaskFailure("Основная фраза должна входить в подтверждённые термины.")
    terms = tuple(QueryTerm(text=item, language="ru" if re.search(r"[А-Яа-яЁё]", item) else
                            "en" if re.search(r"[A-Za-z]", item) else "other",
                            role="technology", origin="user", status="confirmed", confirmed_at=date_confirmed)
                  for item in clean_terms)
    return QueryProfile(profile_id=previous.profile_id if previous is not None else uuid4(),
                        version=previous.version + 1 if previous is not None else 1,
                        original_query=query, definition=definition,
                        exclusions=tuple(normalize_term(item) for item in exclusions), terms=terms,
                        primary_phrase=primary_phrase, confirmed_at=date_confirmed,
                        scientific_plan_hash=previous.scientific_plan_hash if previous is not None else None,
                        supersedes_hash=previous_hash)


@dataclass(frozen=True)
class WordstatMonthlyRequest:
    phrase: str
    from_date: date
    to_date: date
    region_ids: tuple[int, ...]
    devices: tuple[str, ...]
    query_profile_hash: str
    request_hash: str
    matching_mode: str = QUERY_COMPILER_VERSION

    def api_body(self, folder_id: str) -> dict[str, object]:
        """Body for Yandex Search API v2; credentials stay outside the payload."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,50}", folder_id):
            raise TaskFailure("Некорректный идентификатор каталога Yandex Cloud.")
        result: dict[str, object] = {"phrase": self.phrase, "period": "PERIOD_MONTHLY",
                                     "fromDate": self.from_date.isoformat() + "T00:00:00Z",
                                     "toDate": self.to_date.isoformat() + "T00:00:00Z", "folderId": folder_id}
        if self.region_ids:
            result["regions"] = [str(region) for region in self.region_ids]
        if self.devices:
            result["devices"] = ["DEVICE_" + device.upper() for device in self.devices]
        return result


def compile_wordstat_monthly(profile: QueryProfile, policy: SignalPolicy, *,
                             from_date: date, to_date: date) -> tuple[WordstatMonthlyRequest, ...]:
    """Compile only plain, confirmed phrases; other aliases remain visible but unmeasured."""
    if (from_date.day != 1 or from_date < date(2018, 1, 1) or to_date < from_date
            or (to_date + timedelta(days=1)).day != 1):
        raise TaskFailure("Для динамики нужны полные календарные месяцы.")
    if profile.confirmed_at is None or profile.primary_phrase is None:
        raise TaskFailure("Сначала подтвердите основную поисковую фразу.")
    selected = [profile.primary_phrase]
    selected.extend(term.text for term in profile.terms if term.role == "technology" and term.status == "confirmed"
                    and term.text.casefold() != profile.primary_phrase.casefold())
    eligible: list[str] = []
    for phrase in selected:
        if any(character in _OPERATOR_CHARS for character in phrase):
            continue
        if len(eligible) == policy.wordstat_seed_limit:
            break
        eligible.append(phrase)
    if not eligible or eligible[0].casefold() != profile.primary_phrase.casefold():
        raise TaskFailure("Основная фраза содержит оператор Wordstat; выберите простую фразу для месячной истории.")
    profile_hash = object_digest(profile)
    requests = []
    for phrase in eligible:
        payload = {"provider": "wordstat", "method": "dynamics", "version": QUERY_COMPILER_VERSION,
                   "phrase": phrase, "period": "monthly", "fromDate": from_date.isoformat(),
                   "toDate": to_date.isoformat(), "regions": profile.region_ids, "devices": profile.devices,
                   "query_profile_hash": profile_hash}
        requests.append(WordstatMonthlyRequest(phrase, from_date, to_date, profile.region_ids, profile.devices,
                                               profile_hash, content_hash(payload)))
    return tuple(requests)

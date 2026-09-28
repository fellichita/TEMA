"""Сервис веба с засеянными анализами для проверки инструментов владельца без моделей."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np

from app.pilot.approved_sources.contracts import SOURCE_IDS
from app.pilot.publication_relevance import assess_pool
from app.relevance_learning import run_samples, save_run_samples
from app.web_api import RESOURCE_DEFAULTS, WebAnalysisService, _publication_pool
from app.web_monitor import Blocklist, Monitor

AS_OF = date(2026, 9, 24)
TOPICS = {
    "твердотельные аккумуляторы": ("solid-state batteries", ["solid-state battery", "sulfide solid electrolyte",
                                                             "all-solid-state lithium battery"]),
    "квантовые сенсоры": ("quantum sensors", ["quantum sensor", "nitrogen-vacancy magnetometer", "quantum sensing"]),
    "беспилотные грузовики": ("autonomous trucks", ["autonomous truck", "driverless freight", "self-driving truck"]),
}
NOISE = ["Stock market rally lifts shares", "Election debate recap", "Celebrity wedding photos",
         "State of the union speech", "Football transfer rumours", "New smartphone camera review",
         "Weather warning for the weekend", "Housing prices climb again"]
SOURCES = [("google_news", "news_aggregate", "US"), ("google_news", "news_aggregate", "RU"),
           ("semantic_scholar", "journal_article", "INT"), ("cyberleninka", "journal_article", "RU"),
           ("hacker_news", "community", "INT"), ("gdelt", "news_aggregate", "DE"), ("habr", "community", "RU"),
           ("hal", "journal_article", "FR")]


class FakeEncoder:
    """Имитация E5: по теме — косинус 0,88, мимо — 0,70 (детерминированно)."""

    fingerprint = "fake-encoder"

    def __init__(self, words: list[str]):
        self.words = [word.casefold() for word in words]

    def encode(self, texts, *, kind="passage", cancel=None, progress=None):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            if kind == "query":
                vectors[row, 0] = 1
                continue
            similarity = 0.88 if any(word in text.casefold() for word in self.words) else 0.70
            vectors[row, 0] = similarity
            vectors[row, 1] = (1 - similarity ** 2) ** 0.5
        return vectors


def _plan(query: str, english: str, synonyms: list[str]) -> dict[str, Any]:
    return {"original_query": query, "language": "ru", "definition": "d", "english_query": english,
            "subdirections": synonyms[:1], "synonyms": synonyms[1:], "exclusions": [],
            "queries": [], "completed_years": [2020, 2021, 2022, 2023, 2024, 2025],
            "as_of": AS_OF.isoformat(), "planner_version": "test"}


def make_payload(run_number: int, query: str) -> dict[str, Any]:
    english, synonyms = TOPICS[query]
    observations, accepted = [], {}
    for index in range(40):
        source, kind, country = SOURCES[index % len(SOURCES)]
        on_topic = index % 3 != 0
        title = (f"{synonyms[index % len(synonyms)].capitalize()} advance number {index} of run {run_number}"
                 if on_topic else f"{NOISE[index % len(NOISE)]} {index} {run_number}")
        observations.append({"source_id": source, "item_id": f"{run_number}-{index}", "kind": kind, "title": title,
                             "url": f"https://example.org/{run_number}/{index}",
                             "published_at": (AS_OF - timedelta(days=index * 9)).isoformat(),
                             "observed_at": "2026-09-24T00:00:00+00:00", "rights": "local_only",
                             "date_basis": "indexed" if source == "gdelt" else "published", "country": country})
        accepted[source] = accepted.get(source, 0) + 1
    coverage = [{"source_id": source, "state": "complete", "requested_limit": 50, "scanned": accepted.get(source, 0),
                 "accepted": accepted.get(source, 0), "rejected": 0, "duplicates": 0, "limit_reached": False}
                for source in SOURCE_IDS]
    snapshot = {"query": english, "as_of": AS_OF.isoformat(), "collected_at": "2026-09-24T00:00:00+00:00",
                "observations": observations, "coverage": coverage}
    plan = _plan(query, english, synonyms)
    technologies = [{"title": f"{synonyms[position]} platform", "probability": 0.8 - position / 10}
                    for position in range(2 + run_number % 2)]
    return {"result": {"query_plan": plan, "top_trend_ids": ["c1"], "snapshots": [], "quality": "provisional",
                       "cards": [{"candidate": {"candidate_id": "c1", "label": synonyms[0].title(),
                                                "definition": "Definition"}, "category": "early_signal",
                                  "evidence": [{"source_url": "https://example.org/evidence"}]}]},
            "approved_sources": snapshot,
            "radar": {"state": "ready", "result": {"technologies": technologies, "excluded": []}},
            "source_policy": {"version": "source-policy/1", "countries": ["RU"] if run_number % 2 else [],
                              "disabled": []}}


class OwnerPilot:
    def __init__(self, data_dir: Path, runs: int = 6):
        self.data_dir = data_dir
        self.archive = None
        self._payloads: dict[str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        self._lock = Lock()
        self.coordinator = self
        queries = list(TOPICS)
        for number in range(runs):
            run_id = f"run-{number + 1}"
            query = queries[number % len(queries)]
            payload = make_payload(number, query)
            from app.pilot.approved_sources import SourceSnapshot

            snapshot = SourceSnapshot.model_validate(payload["approved_sources"])
            pool = _publication_pool(payload["result"], snapshot, None, keep_studies=True)
            english, synonyms = TOPICS[query]
            relevance = assess_pool(pool, payload["result"]["query_plan"],
                                    encoder=FakeEncoder([english, *synonyms]))
            payload["publication_relevance"] = relevance
            save_run_samples(data_dir, run_samples(
                run_id, payload["result"]["query_plan"], pool,
                relevance={key: {"cosine": item[3]} for key, item in relevance["items"].items()},
                created_at=(datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=number)).isoformat()))
            self._payloads[run_id] = payload
            created = datetime(2026, 9, 1, 9, tzinfo=UTC) + timedelta(days=number)
            state = "failed" if number == runs - 1 and runs > 3 else "succeeded"
            self._rows.insert(0, {"id": run_id, "state": state, "attempt": 1,
                                  "error": "Сбой источника" if state == "failed" else None,
                                  "created_at": created.isoformat(),
                                  "updated_at": (created + timedelta(minutes=3 + number)).isoformat(),
                                  "input_json": json.dumps({"payload": {
                                      "query": query, "collection_profile": "deep" if number % 3 == 2 else "fast",
                                      "settings": {"provider": "local"}}})})

    # PilotService
    def list_runs(self, offset=0, limit=50, source="local"):
        return [dict(row) for row in self._rows[offset:offset + limit]]

    def get(self, run_id):
        row = next(row for row in self._rows if row["id"] == run_id)
        return dict(row)

    def result(self, run_id):
        if self.get(run_id)["state"] != "succeeded":
            raise ValueError("not finished")
        return json.loads(json.dumps(self._payloads[run_id]))

    def status(self):
        return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"}, "keys": {}}

    # Coordinator
    def checkpoint_value(self, run_id, stage, cancel=None):
        return None

    def running(self):
        return []

    slots = 2


def make_service(data_dir: Path, runs: int = 6) -> WebAnalysisService:
    service = object.__new__(WebAnalysisService)
    service.pilot = OwnerPilot(data_dir, runs)
    service.poll_seconds, service.timeout_seconds = 0.1, 60
    for name in ("_admission", "_state_lock", "_rendered_lock", "_translation_lock", "_reader_lock", "_radar_lock"):
        setattr(service, name, Lock())
    service._rendered, service._translations, service._radars = {}, {}, {}
    service._reader = None
    service._current, service._cancel_requested_id = None, None
    service.separate_visitors = False
    service._runs, service._owners, service._latest = {}, {}, {}
    service.monitor = Monitor(anonymous="Локальный пользователь", blocklist=Blocklist(None))
    service.paused = False
    service._closing = Event()
    service._admin_status = None
    service._queue, service._cancelled, service._dispatching = [], set(), False
    service.resources = dict(RESOURCE_DEFAULTS)
    service._resources_path = None
    return service

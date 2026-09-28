"""Инструменты владельца в панели управления: источники, языки, анализ анализов, обучение.

Методы подмешиваются в сервис веба (`WebAnalysisService`) и вызываются только
через API владельца с отдельным токеном; посетителям сайта они не видны.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
import re
from threading import Lock, Thread
from typing import Any

RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
PUBLICATION_ID = re.compile(r"[0-9a-f]{64}\Z")
MAX_COMPARE = 6
DECISION_FILTERS = frozenset({"all", "relevant", "weak", "off_topic", "marked", "unrated"})
BACKFILL_RUNS = 60
POOL_CACHE = 2


def _mapping(value: object) -> dict[str, Any]:
    """Словарь из разобранного JSON или пустой словарь."""
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    """Список из разобранного JSON или пустой список."""
    return value if isinstance(value, list) else []


def _error(status: int, message: str) -> Exception:
    from app.web_api import WebApiError

    return WebApiError(status, message)


class OwnerTools:
    """Вкладки панели владельца поверх сервиса анализа."""

    pilot: Any

    # --- общее ---------------------------------------------------------------------------------

    def _owner_data_dir(self) -> Path:
        data_dir = getattr(getattr(self, "pilot", None), "data_dir", None)
        if not isinstance(data_dir, Path):
            raise _error(HTTPStatus.SERVICE_UNAVAILABLE, "Профиль данных сервиса недоступен.")
        return data_dir

    def _owner_event(self, text: str, level: str = "info") -> None:
        monitor = getattr(self, "_monitor", None)
        if callable(monitor):
            try:
                monitor().event("owner", text, level=level)
            except Exception:
                pass

    # --- источники и страны --------------------------------------------------------------------

    def source_policy(self) -> Any:
        """Правило владельца и, при включённой адаптации, доли, выученные на прошлых анализах."""
        from app.pilot.approved_sources.catalog import SOURCE_INFO, SourcePolicy, load_policy
        from app.relevance_learning import RelevanceModel, load_settings

        data_dir = getattr(getattr(self, "pilot", None), "data_dir", None)
        if not isinstance(data_dir, Path):
            return SourcePolicy()
        policy = load_policy(data_dir)
        weights: dict[str, float] = {}
        try:
            if load_settings(data_dir)["adapt_sources"]:
                weights = {source: weight for source, weight in RelevanceModel.load(data_dir).source_weights().items()
                           if source in SOURCE_INFO}
        except Exception:
            weights = {}
        return SourcePolicy(policy.countries, policy.disabled, weights)

    def start_options(self) -> dict[str, Any]:
        """Дополнительные параметры запуска анализа; пусто, если правило по умолчанию."""
        from app.pilot.approved_sources.catalog import SourcePolicy

        policy = self.source_policy()
        return {} if policy == SourcePolicy() else {"source_policy": policy.to_json()}

    def admin_sources(self) -> dict[str, Any]:
        from app.pilot.approved_sources.catalog import COUNTRIES, catalogue

        policy = self.source_policy()
        return {"countries": [{"code": code, "name": name, "selected": code in policy.countries}
                              for code, name in COUNTRIES.items()],
                "everywhere": policy.everywhere, "policy": policy.to_json(), "sources": catalogue(policy)}

    def set_sources(self, values: object) -> dict[str, Any]:
        from app.pilot.approved_sources.catalog import SourcePolicy, load_policy, save_policy

        if not isinstance(values, dict) or not set(values) <= {"countries", "disabled"} or not values:
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
        data_dir = self._owner_data_dir()
        current = load_policy(data_dir)
        try:
            policy = SourcePolicy.from_json({"countries": values.get("countries", list(current.countries)),
                                             "disabled": values.get("disabled", list(current.disabled))})
        except (TypeError, ValueError):
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "Неизвестная страна или источник.") from None
        from app.pilot.approved_sources.catalog import SOURCE_INFO

        if len(policy.disabled) == len(SOURCE_INFO) or not any(policy.allows(source) for source in SOURCE_INFO):
            raise _error(HTTPStatus.CONFLICT, "При таком выборе не остаётся ни одного источника.")
        save_policy(data_dir, policy)
        countries = ", ".join(policy.countries) if policy.countries else "все страны"
        self._owner_event(f"Владелец изменил источники: {countries}; выключено {len(policy.disabled)}")
        return {"message": "Источники обновлены. Правило действует для новых анализов.",
                "sources": self.admin_sources()}

    # --- языки ---------------------------------------------------------------------------------

    def _language_manager(self) -> Any:
        manager = self.__dict__.get("_language_store")
        if manager is None:
            from app.pilot.languages import LanguageManager

            manager = self.__dict__.setdefault("_language_store", LanguageManager(self._owner_data_dir()))
        return manager

    def admin_languages(self) -> dict[str, Any]:
        return self._language_manager().overview()

    def set_languages(self, values: object) -> dict[str, Any]:
        if not isinstance(values, dict) or values.get("op") not in {"add", "install", "remove", "default"} \
                or not set(values) <= {"op", "code", "name", "model_id"}:
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
        try:
            message = self._language_manager().act(values)
        except ValueError as error:
            raise _error(HTTPStatus.CONFLICT, str(error) or "Действие с языком недоступно.") from None
        if values["op"] == "remove":
            readers = self.__dict__.get("_readers")
            if isinstance(readers, dict):
                readers.pop(values.get("code"), None)
        self._owner_event("Владелец: " + message)
        return {"message": message, "languages": self.admin_languages()}

    # --- самообучение --------------------------------------------------------------------------

    def _trainer(self) -> Any:
        trainer = self.__dict__.get("_relevance_trainer")
        if trainer is None:
            from app.relevance_learning import Trainer

            trainer = self.__dict__.setdefault("_relevance_trainer", Trainer(self._owner_data_dir()))
        return trainer

    def learn_after_run(self, _run_id: str | None = None) -> None:
        """После успешного анализа — переобучение в фоне, если владелец его не выключил."""
        from app.relevance_learning import load_settings

        try:
            settings = load_settings(self._owner_data_dir())
            if settings["enabled"] and settings["auto_retrain"]:
                self._start_training(backfill=False)
        except Exception:
            pass

    def _start_training(self, *, backfill: bool) -> bool:
        trainer = self._trainer()
        state = self.__dict__.setdefault("_learning_state", {"backfilled": 0, "backfill_running": False})

        def run(target: Any) -> None:
            def work() -> None:
                if backfill:
                    state["backfill_running"] = True
                    try:
                        state["backfilled"] += self._backfill_samples(BACKFILL_RUNS)
                    finally:
                        state["backfill_running"] = False
                target()
            Thread(target=work, daemon=True, name="relevance-training").start()

        return trainer.request(start=run)

    def _backfill_samples(self, limit: int) -> int:
        """Примеры для обучения из прошлых анализов, сделанных до самообучения."""
        from app.pilot.approved_sources import SourceSnapshot
        from app.relevance_learning import discovery_decisions, run_samples, sampled_runs, save_run_samples
        from app.web_api import _publication_pool, history_entry

        data_dir = self._owner_data_dir()
        known = sampled_runs(data_dir)
        added = 0
        for page in range(0, 20):
            rows = self.pilot.list_runs(offset=page * 50, limit=50, source="local")
            for row in rows:
                if added >= limit:
                    return added
                entry = history_entry(row)
                if entry is None or entry["state"] != "succeeded" or entry["id"] in known:
                    continue
                try:
                    payload = self.pilot.result(entry["id"])
                    result = payload["result"]
                    snapshot = (SourceSnapshot.model_validate(payload["approved_sources"])
                                if isinstance(payload.get("approved_sources"), dict) else None)
                    pool = _publication_pool(result, snapshot, getattr(self.pilot, "archive", None),
                                             keep_studies=True)
                    candidates = self.pilot.coordinator.checkpoint_value(entry["id"], "candidates")
                    relevance = payload.get("publication_relevance") if isinstance(
                        payload.get("publication_relevance"), dict) else {}
                    save_run_samples(data_dir, run_samples(
                        entry["id"], result.get("query_plan") or {}, pool,
                        relevance={pid: {"cosine": item[3]} for pid, item in (relevance.get("items") or {}).items()
                                   if isinstance(item, list) and len(item) >= 4},
                        discovery=discovery_decisions((candidates or {}).get("relevance") or ()),
                        created_at=entry["created_at"]))
                    added += 1
                except Exception:
                    continue
            if len(rows) < 50:
                break
        return added

    def admin_learning(self) -> dict[str, Any]:
        from app.relevance_learning import summary

        data = summary(self._owner_data_dir())
        trainer = self._trainer()
        state = self.__dict__.get("_learning_state", {})
        data.update(training=trainer.running, last_error=trainer.last_error,
                    backfill_running=bool(state.get("backfill_running")), backfilled=state.get("backfilled", 0))
        return data

    def learning_action(self, action: str, values: object) -> dict[str, Any]:
        from app.relevance_learning import reset, save_settings

        data_dir = self._owner_data_dir()
        if action == "learning":
            try:
                settings = save_settings(data_dir, values if isinstance(values, dict) else {})
            except ValueError:
                raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request") from None
            self._owner_event("Владелец изменил настройки самообучения: " + ", ".join(
                f"{name} = {'да' if value else 'нет'}" for name, value in settings.items()))
            return {"message": "Настройки обучения сохранены.", "learning": self.admin_learning()}
        if action == "retrain":
            backfill = isinstance(values, dict) and values.get("backfill") is True
            started = self._start_training(backfill=backfill)
            self._owner_event("Владелец запустил переобучение модели" + (" на прошлых анализах" if backfill else ""))
            return {"message": "Переобучение запущено." if started else "Обучение уже идёт: повторим после него."}
        if action == "reset_learning":
            if self._trainer().running:
                raise _error(HTTPStatus.CONFLICT, "Дождитесь окончания обучения.")
            reset(data_dir, keep_feedback=not (isinstance(values, dict) and values.get("feedback") is True))
            self._owner_event("Владелец сбросил обученную модель", level="warning")
            return {"message": "Модель и примеры забыты; анализы идут на эвристике до нового обучения."}
        if action == "feedback":
            return self.set_feedback(values)
        raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")

    def set_feedback(self, values: object) -> dict[str, Any]:
        """Отметка владельца «по теме / не по теме» у публикации анализа — лучший учитель модели."""
        from app.relevance_learning import set_feedback

        if (not isinstance(values, dict) or set(values) != {"run_id", "publication_id", "label"}
                or not isinstance(values["run_id"], str) or RUN_ID.fullmatch(values["run_id"]) is None
                or not isinstance(values["publication_id"], str)
                or PUBLICATION_ID.fullmatch(values["publication_id"]) is None
                or values["label"] not in {0, 1, None} or isinstance(values["label"], bool)):
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
        item = next((publication for publication in self._run_pool(values["run_id"])[0]
                     if publication["publication_id"] == values["publication_id"]), None)
        if item is None:
            raise _error(HTTPStatus.NOT_FOUND, "Публикация не найдена в этом анализе.")
        answer = set_feedback(self._owner_data_dir(), values["run_id"], item, values["label"])
        return {"message": {1: "Отмечено: по теме.", 0: "Отмечено: не по теме.", None: "Отметка снята."}[
            values["label"]], **answer}

    # --- анализ анализов -----------------------------------------------------------------------

    def _index(self) -> Any:
        index = self.__dict__.get("_analysis_index")
        if index is None:
            from app.analysis_index import AnalysisIndex

            index = self.__dict__.setdefault("_analysis_index", AnalysisIndex(self._owner_data_dir()))
        return index

    def _index_rows(self) -> list[dict[str, Any]]:
        from app.analysis_index import MAX_RUNS

        rows: list[dict[str, Any]] = []
        for page in range(MAX_RUNS // 50):
            batch = self.pilot.list_runs(offset=page * 50, limit=50, source="local")
            rows.extend(batch)
            if len(batch) < 50:
                break
        return rows

    def _index_entry(self, row: Any) -> dict[str, Any] | None:
        from app.web_api import history_entry

        entry = history_entry(row)
        if entry is None:
            return None
        owner = self.__dict__.get("_owners", {}).get(entry["id"])
        monitor = getattr(self, "_monitor", None)
        entry["visitor"] = monitor().label(owner) if owner is not None and callable(monitor) else None
        return entry

    def _index_summary(self, run_id: str) -> dict[str, Any] | None:
        from app.analysis_index import result_summary

        return result_summary(self.pilot.result(run_id))

    def admin_analyses(self, parameters: dict[str, str]) -> dict[str, Any]:
        from app.analysis_index import query_entries, valid_filters

        try:
            filters = valid_filters(parameters)
            offset = int(parameters.get("offset", "0"))
            limit = int(parameters.get("limit", "50"))
            sort = parameters.get("sort", "created_at")
            descending = parameters.get("order", "desc") != "asc"
            if not 0 <= offset <= 100_000 or not 1 <= limit <= 200:
                raise ValueError("page")
        except ValueError:
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректный фильтр анализов.") from None
        index = self._index()
        progress = index.refresh(self._index_rows(), self._index_entry, self._index_summary)
        try:
            answer = query_entries(index.entries(), filters=filters, sort=sort, descending=descending,
                                   offset=offset, limit=limit)
        except ValueError:
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная сортировка.") from None
        return {**answer, "index": progress, "filters": filters, "sort": sort,
                "order": "desc" if descending else "asc"}

    def admin_compare(self, ids: list[str]) -> dict[str, Any]:
        from app.analysis_index import compare

        if not 2 <= len(ids) <= MAX_COMPARE or len(set(ids)) != len(ids) or any(
                RUN_ID.fullmatch(identifier) is None for identifier in ids):
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "Выберите от 2 до 6 анализов.")
        index = self._index()
        index.refresh(self._index_rows(), self._index_entry, self._index_summary, budget=len(ids))
        known = {item["id"]: item for item in index.entries()}
        missing = [identifier for identifier in ids if identifier not in known]
        if missing:
            raise _error(HTTPStatus.NOT_FOUND, "Часть выбранных анализов не найдена.")
        return compare([known[identifier] for identifier in ids])

    def _run_pool(self, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
        """Весь пул публикаций анализа (включая отсеянные) и оценки темы.

        Готовый анализ не меняется, поэтому пул двух последних открытых анализов
        держится в памяти: листание и отметки не перечитывают архив документов.
        """
        cache = self.__dict__.setdefault("_owner_pools", {})
        with self.__dict__.setdefault("_owner_pool_lock", Lock()):
            cached = cache.pop(run_id, None)
            if cached is None:
                cached = self._read_run_pool(run_id)
            cache[run_id] = cached
            while len(cache) > POOL_CACHE:
                del cache[next(iter(cache))]
            return cached

    def _read_run_pool(self, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
        from app.pilot.approved_sources import SourceSnapshot
        from app.pilot.publication_relevance import relevance_items
        from app.web_api import _publication_pool

        try:
            payload = self.pilot.result(run_id)
        except Exception:
            raise _error(HTTPStatus.NOT_FOUND, "Анализ не найден или ещё не завершён.") from None
        result = payload.get("result")
        if not isinstance(result, dict):
            raise _error(HTTPStatus.NOT_FOUND, "Анализ не найден или ещё не завершён.")
        snapshot = (SourceSnapshot.model_validate(payload["approved_sources"])
                    if isinstance(payload.get("approved_sources"), dict) else None)
        pool = _publication_pool(result, snapshot, getattr(self.pilot, "archive", None))
        relevance = _mapping(payload.get("publication_relevance"))
        summary = {"publication_relevance": {key: relevance.get(key) for key in ("model_version", "semantic")}}
        return pool, summary, relevance_items(payload.get("publication_relevance"))

    def admin_publications(self, parameters: dict[str, str]) -> dict[str, Any]:
        """Публикации анализа с оценкой темы и отметками владельца — для проверки и обучения."""
        from app.relevance_learning import load_feedback

        run_id = parameters.get("run_id", "")
        decision = parameters.get("decision", "all")
        try:
            offset, limit = int(parameters.get("offset", "0")), int(parameters.get("limit", "30"))
        except ValueError:
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request") from None
        if (RUN_ID.fullmatch(run_id) is None or decision not in DECISION_FILTERS
                or not 0 <= offset <= 1_000_000 or not 1 <= limit <= 100):
            raise _error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
        pool, payload, relevance = self._run_pool(run_id)
        marks = load_feedback(self._owner_data_dir()).get(run_id, {})
        rows = []
        counts = {"relevant": 0, "weak": 0, "off_topic": 0, "unrated": 0, "marked": 0}
        for publication in pool:
            entry = relevance.get(publication["publication_id"]) if relevance is not None else None
            state = entry[1] if entry is not None else "unrated"
            counts[state] += 1
            mark = marks.get(publication["publication_id"])
            if mark is not None:
                counts["marked"] += 1
            if decision == "marked" and mark is None or decision not in {"all", "marked"} and state != decision:
                continue
            rows.append({"publication_id": publication["publication_id"], "title": publication["title"],
                         "summary": (publication.get("summary") or "")[:300] or None,
                         "url": publication["url"], "source_id": publication["source_id"],
                         "country": publication.get("country"), "published_at": publication.get("published_at"),
                         "publication_year": publication.get("publication_year"),
                         "decision": state, "score": entry[0] if entry is not None else None,
                         "lexical": entry[2] if entry is not None else None,
                         "cosine": entry[3] if entry is not None else None,
                         "mark": mark.get("label") if isinstance(mark, dict) else None})
        rows.sort(key=lambda row: (-(row["score"] if row["score"] is not None else -1), row["title"]))
        relevance_payload = _mapping(payload.get("publication_relevance"))
        return {"run_id": run_id, "total": len(rows), "offset": offset, "counts": counts,
                "rated": relevance is not None, "model_version": relevance_payload.get("model_version"),
                "semantic": relevance_payload.get("semantic"), "publications": rows[offset:offset + limit]}

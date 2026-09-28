"""Read saved JSON or a finished backend history without changing source data."""

from concurrent.futures import CancelledError
import calendar
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from app.backend.contracts import normalize_doi
from app.input_safety import MAX_ABSTRACT_CHARACTERS, MAX_TITLE_CHARACTERS
from app.ml.contracts import AnalysisInputError


MAX_JSON_DEPTH = 128


def checkpoint(cancel=None):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Анализ отменён.")


def _finite_json_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON-снимок содержит неограниченное или нечисловое значение.")
    return number


def read_snapshot(path, max_documents=20_000, cancel=None):
    checkpoint(cancel)
    path = Path(path)
    if path.stat().st_size > 300_000_000:
        raise AnalysisInputError("Снимок превышает лимит 300 МБ.")
    payload = path.read_bytes()
    checkpoint(cancel)
    try:
        data = json.loads(payload, parse_float=_finite_json_number, parse_constant=_finite_json_number)
    except RecursionError as error:
        raise AnalysisInputError(f"Вложенность JSON-снимка превышает лимит {MAX_JSON_DEPTH} уровней.") from error
    except (ValueError, UnicodeError) as error:
        raise AnalysisInputError("Не удалось прочитать JSON-снимок корпуса.") from error
    result = unpack_snapshot(data, max_documents, cancel)
    result["provenance"]["snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def read_history(backend, history_id, max_documents=20_000, cancel=None):
    history = backend.get_history(history_id)
    if history.state in {"running", "queued"}:
        raise AnalysisInputError("Дождитесь окончания исторического сбора.")
    batches, count = [], 0
    for period in history.periods:
        if period.state == "split" or period.job is None:
            continue
        records, offset = [], 0
        while True:
            checkpoint(cancel)
            page = backend.list_documents(job_id=period.job.id, limit=500, offset=offset)
            records.extend(item.model_dump(mode="json") for item in page.items)
            offset += len(page.items)
            if count + offset > max_documents:
                raise AnalysisInputError(f"В корпусе больше {max_documents} записей. Выберите более узкое направление.")
            if offset >= page.total:
                break
            if not page.items:
                raise AnalysisInputError("Неполный экспорт документов истории.")
        count += len(records)
        batches.append({"job_id": period.job.id, "total": page.total, "documents": records})
    return unpack_snapshot({"schema_version": 1, "history": history.model_dump(mode="json"), "batches": batches},
                           max_documents, cancel)


def _require(condition, field):
    if not condition:
        raise AnalysisInputError(f"Некорректное поле {field} в снимке.")


def _text(value, field, nonempty=False):
    _require(isinstance(value, str) and (not nonempty or bool(value.strip())), field)


def _check_json_depth(data, cancel=None):
    """Iterative depth check also covers otherwise unconsumed provider metadata."""
    stack = [iter((data,))]
    visited = 0
    while stack:
        if visited % 1024 == 0:
            checkpoint(cancel)
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visited += 1
        if isinstance(value, (dict, list)):
            if len(stack) > MAX_JSON_DEPTH:
                raise AnalysisInputError(f"Вложенность JSON-снимка превышает лимит {MAX_JSON_DEPTH} уровней.")
            stack.append(iter(value.values() if isinstance(value, dict) else value))


def _string_list(value, field):
    _require(isinstance(value, list) and all(isinstance(item, str) for item in value), field)


def normalize_snapshot_doi(value):
    """Canonical identity for accepted legacy DOI aliases, without editing records."""
    value = value.strip()
    legacy_prefix = "http://dx.doi.org/"
    if value.lower().startswith(legacy_prefix):
        value = "https://dx.doi.org/" + value[len(legacy_prefix):]
    return normalize_doi(value)


def _snapshot_date(value, field):
    _text(value, field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise AnalysisInputError(f"Некорректная дата {field} в снимке.") from error
    _require(parsed.isoformat() == value, field)
    return parsed


def _request_fields(request):
    _require(isinstance(request, dict), "request")
    _text(request["topic"], "request.topic", nonempty=True)
    if "primary_topic_ids" in request:
        _string_list(request["primary_topic_ids"], "request.primary_topic_ids")
    dates = {key: _snapshot_date(request[key], f"request.{key}")
             for key in ("from_date", "until_date") if request.get(key) is not None}
    if len(dates) == 2:
        _require(dates["from_date"] <= dates["until_date"], "request.from_date/until_date")


def _publication_interval(document):
    """Bounds represent uncertainty only; never add dates to the source record."""
    year, month, day = (document.get(key) for key in
                        ("publication_year", "publication_month", "publication_date"))
    if year is not None:
        _require(type(year) is int and 1 <= year <= 9999, "document.publication_year")
    if month is not None:
        _require(type(month) is int and 1 <= month <= 12, "document.publication_month")
        _require(year is not None, "document.publication_year")
    parsed = _snapshot_date(day, "document.publication_date") if day is not None else None
    if parsed is not None:
        if parsed.year != year:
            raise AnalysisInputError("Год и дата документа не совпадают.")
        _require(month is None or month == parsed.month, "document.publication_month")
    precision = document.get("date_precision")
    if "date_precision" in document:
        _require(isinstance(precision, str) and precision in {"day", "month", "year", "unknown"},
                 "document.date_precision")
        consistent = {"day": parsed is not None,
                      "month": year is not None and month is not None and parsed is None,
                      "year": year is not None and month is None and parsed is None,
                      "unknown": year is None and month is None and parsed is None}
        _require(consistent[precision], "document.date_precision")
    # Legacy snapshots may omit precision while preserving the available fields.
    if parsed is not None:
        return parsed, parsed
    if month is not None:
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    if year is not None:
        return date(year, 1, 1), date(year, 12, 31)
    return None


def _document_fields(entry, source):
    """Validate consumed JSON types without coercion or backend-model defaults."""
    _require(isinstance(entry, dict), "entry")
    for key in ("document_key", "revision_id"):
        _text(entry[key], key, nonempty=True)
    document = entry["document"]
    _require(isinstance(document, dict), "document")
    _text(document.get("title"), "document.title")
    _require(document.get("source") == source, "document.source")
    for key in ("abstract", "doi", "source_id", "url", "document_type", "language", "fetched_at"):
        if document.get(key) is not None:
            _text(document[key], f"document.{key}")
    for key, limit in (("title", MAX_TITLE_CHARACTERS), ("abstract", MAX_ABSTRACT_CHARACTERS)):
        if document.get(key) is not None and len(document[key]) > limit:
            raise AnalysisInputError(f"Поле document.{key} превышает лимит {limit} символов.")
    if document.get("doi") is not None:
        try:
            normalize_snapshot_doi(document["doi"])
        except ValueError as error:
            raise AnalysisInputError("Некорректный DOI документа в снимке.") from error
    if document.get("authors") is not None:
        _string_list(document["authors"], "document.authors")
    if document.get("raw_metadata") is not None:
        _require(isinstance(document["raw_metadata"], dict), "document.raw_metadata")
        retracted = document["raw_metadata"].get("is_retracted")
        _require(retracted is None or type(retracted) is bool, "document.raw_metadata.is_retracted")


def _collection_quality(period, job, batch, documents):
    """Only an evidenced invalid-record loss can soften a partial collection."""
    scanned, stored, skipped = (job.get(key) for key in ("scanned", "stored", "skipped"))
    total = job.get("total_available")
    proven = (
        period["state"] == "partial"
        and period.get("incomplete_reason") == "invalid_records_skipped"
        and job.get("state") == "succeeded"
        and job.get("source_exhausted") is True
        and all(type(value) is int for value in (scanned, stored, skipped))
        and 0 < skipped <= scanned - stored
        and (total is None or scanned >= total)
        and batch is not None
        and batch.get("total") == stored == len(documents)
    )
    if not proven:
        return [], []
    # Backend deduplicates source identities, so stored + skipped may be < scanned.
    issues = sorted(["Сбор периода неполный", "Выдача не исчерпана или есть пропуски"])
    note = ("Выдача источника исчерпана, сохранённые документы экспортированы полностью; "
            "partial вызван только пропуском невалидных записей "
            f"(skipped={skipped}, scanned={scanned}, stored={stored}, export={len(documents)}, "
            f"total_available={total}). Динамика относится к пригодной сохранённой выборке; "
            "содержание пропущенных записей неизвестно.")
    return issues, [note]


def unpack_snapshot(data, max_documents=20_000, cancel=None):
    _check_json_depth(data, cancel)
    try:
        _require(isinstance(data, dict), "snapshot")
        history, batches = data["history"], data["batches"]
        _require(isinstance(history, dict), "history")
        _require(isinstance(batches, list), "batches")
        schema_version, contract_version = data.get("schema_version"), history.get("contract_version", 1)
        if (type(schema_version) is not int or schema_version != 1 or type(contract_version) is not int
                or contract_version not in (1, 2)):
            raise AnalysisInputError("Неподдерживаемая версия снимка.")
        _text(history["id"], "history.id", nonempty=True)
        _text(history["state"], "history.state", nonempty=True)
        _require(isinstance(history["periods"], list), "history.periods")
        if history["state"] in {"running", "queued"}:
            raise AnalysisInputError("Исторический сбор ещё не завершён.")
        _request_fields(history["request"])
        sources = history["request"]["sources"]
        _string_list(sources, "request.sources")
        if len(sources) != 1 or sources[0] not in {"openalex", "crossref"}:
            raise AnalysisInputError("Для MVP выберите историю одного научного источника: OpenAlex или Crossref.")
        by_job = {}
        for batch in batches:
            _require(isinstance(batch, dict), "batch")
            _text(batch["job_id"], "batch.job_id", nonempty=True)
            _require(isinstance(batch["documents"], list), "batch.documents")
            if "total" in batch:
                _require(type(batch["total"]) is int and batch["total"] >= 0, "batch.total")
            if batch["job_id"] in by_job:
                raise AnalysisInputError("В снимке повторена партия одного задания.")
            by_job[batch["job_id"]] = batch
        entries, periods = [], []
        for period in history["periods"]:
            checkpoint(cancel)
            _require(isinstance(period, dict), "period")
            _text(period["state"], "period.state", nonempty=True)
            period_start = _snapshot_date(period["from_date"], "period.from_date")
            period_end = _snapshot_date(period["until_date"], "period.until_date")
            _require(period_start <= period_end, "period.from_date/until_date")
            if period["state"] == "split":
                continue
            if period.get("job") is None:
                if period["state"] == "complete":
                    raise AnalysisInputError("Для завершённого периода отсутствует задание сбора.")
                periods.append({"from_date": period["from_date"], "until_date": period["until_date"],
                                "issues": ["Сбор периода неполный", "Для периода не создано задание"],
                                "nonblocking_issues": [], "data_quality_notes": [], "undated_records": 0})
                continue
            job = period["job"]
            _require(isinstance(job, dict), "period.job")
            _text(job["id"], "job.id", nonempty=True)
            if "state" in job:
                _text(job["state"], "job.state", nonempty=True)
            for key in ("scanned", "stored", "skipped", "total_available"):
                if key in job and not (key == "total_available" and job[key] is None):
                    _require(type(job[key]) is int and job[key] >= 0, f"job.{key}")
            if "source_exhausted" in job:
                _require(type(job["source_exhausted"]) is bool, "job.source_exhausted")
            batch = by_job.get(job.get("id"))
            documents = batch["documents"] if batch else []
            issues = []
            if period["state"] != "complete" or job.get("state") != "succeeded":
                issues.append("Сбор периода неполный")
            if not job.get("source_exhausted") or job.get("skipped") != 0:
                issues.append("Выдача не исчерпана или есть пропуски")
            if batch is None or batch.get("total") != len(documents) or job.get("stored") != len(documents):
                issues.append("Неполный экспорт периода")
            counters = [job.get(key) for key in ("scanned", "stored", "skipped")]
            if any(value is None for value in counters):
                issues.append("Недостаточно данных о счётчиках сбора периода")
            elif counters[1] + counters[2] > counters[0]:
                issues.append("Несогласованные счётчики сбора периода")
            if job.get("total_available") is not None and job.get("scanned", 0) < job["total_available"]:
                issues.append("Счётчик источника не достигнут")
            request = job.get("request") or {}
            _request_fields(request)
            expected = {"topic": history["request"]["topic"], "source": sources[0],
                        "from_date": period["from_date"], "until_date": period["until_date"]}
            if any(request.get(k) != v for k, v in expected.items()) or (
                    request.get("primary_topic_ids", []) != history["request"].get("primary_topic_ids", [])):
                raise AnalysisInputError("Запрос задания не соответствует истории.")
            undated_records = 0
            for entry in documents:
                checkpoint(cancel)
                _document_fields(entry, sources[0])
                d = entry["document"]
                interval = _publication_interval(d)
                if interval is None:
                    undated_records += 1
                    issues.append("Год публикации неизвестен; принадлежность периоду не подтверждена.")
                elif interval[1] < period_start or interval[0] > period_end:
                    issues.append("Документы вне периода запроса")
                entries.append(entry)
            if len(entries) > max_documents:
                raise AnalysisInputError(f"В корпусе больше {max_documents} записей. Выберите более узкое направление.")
            nonblocking, quality_notes = _collection_quality(period, job, batch, documents)
            periods.append({"from_date": period["from_date"], "until_date": period["until_date"],
                            "issues": sorted(set(issues)), "nonblocking_issues": nonblocking,
                            "data_quality_notes": quality_notes, "undated_records": undated_records})
        return {"entries": entries, "periods": periods, "topic": history["request"]["topic"],
                "source": sources[0], "provenance": {"history_id": history["id"],
                "history_state": history["state"], "request": history["request"]}}
    except (KeyError, TypeError, AttributeError) as error:
        raise AnalysisInputError("Ожидается JSON исторического корпуса с полями history и batches.") from error

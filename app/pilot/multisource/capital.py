"""Project-level grants and explicit investment CSVs, without monetary double count."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
from typing import Literal, cast
from uuid import NAMESPACE_URL, uuid5

from app.pilot.contracts import content_hash
from app.pilot.multisource.contracts import (CapitalDescription, CapitalEvent, CapitalImportReceipt, ExportRight,
                                             QueryProfile, SourceSnapshot, TechnologyAssociation, TechnologyConcept,
                                             validate_import_export_right)
from app.pilot.multisource.imports import CsvDelimiter, CsvDocument, CsvEncoding, read_csv_document
from app.pilot.multisource.store import SignalStore, object_digest
from app.runtime.jobs import TaskFailure

MoneyFormat = Literal["decimal_comma", "decimal_dot"]
DateFormat = Literal["YYYY-MM-DD", "DD.MM.YYYY", "YYYY-MM", "YYYY"]


@dataclass(frozen=True)
class CordisMapping:
    project_id_column: str
    title_column: str
    objective_column: str
    ec_contribution_column: str
    ec_signature_column: str
    money_format: MoneyFormat
    signature_format: Literal["YYYY-MM-DD", "DD.MM.YYYY"]
    programme_column: str | None = None
    coordinator_column: str | None = None


@dataclass(frozen=True)
class ParticipantMapping:
    project_id_column: str
    organisation_id_column: str
    role_column: str | None = None


@dataclass(frozen=True)
class InvestmentMapping:
    event_id_column: str
    recipient_id_column: str
    recipient_name_column: str
    kind_column: str
    status_column: str
    date_column: str
    date_format: DateFormat
    amount_column: str
    currency_column: str
    money_format: MoneyFormat
    description_column: str | None = None
    url_column: str | None = None


@dataclass(frozen=True)
class CapitalParse:
    events: tuple[CapitalEvent, ...]
    descriptions: tuple[CapitalDescription, ...]
    rejected: tuple[str, ...]


def _money(value: str, number_format: MoneyFormat) -> str | None:
    if not value:
        return None
    if number_format not in {"decimal_comma", "decimal_dot"}:
        raise ValueError("money_format")
    separator = "," if number_format == "decimal_comma" else "."
    other = "." if separator == "," else ","
    if other in value:
        raise ValueError("ambiguous_money_separator")
    pieces = value.split(separator)
    if len(pieces) > 2 or len(pieces) == 2 and not 1 <= len(pieces[1]) <= 2:
        raise ValueError("money_precision")
    whole = pieces[0].replace("\u00a0", " ").replace("\u202f", " ")
    if not (re.fullmatch(r"\d{1,20}", whole) or
            re.fullmatch(r"[1-9]\d{0,2}(?: \d{3})+", whole)):
        raise ValueError("money_digits")
    normalized = whole.replace(" ", "") + ("." + pieces[1] if len(pieces) == 2 else "")
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        raise ValueError("money_digits") from None
    if not number.is_finite() or number < 0:
        raise ValueError("money_range")
    return format(number, "f")


def _date(value: str, fmt: DateFormat) -> tuple[date | None, str | None, int | None]:
    if not value:
        return None, None, None
    if fmt == "YYYY-MM-DD" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value), None, None
    if fmt == "DD.MM.YYYY" and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
        day, month, year = (int(item) for item in value.split("."))
        return date(year, month, day), None, None
    if fmt == "YYYY-MM" and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        return None, value, None
    if fmt == "YYYY" and re.fullmatch(r"\d{4}", value):
        return None, None, int(value)
    raise ValueError("event_date")


def _cell(document: CsvDocument, row: tuple[str, ...], column: str | None) -> str | None:
    return row[document.column(column)] if column is not None else None


def _participants(document: CsvDocument | None, mapping: ParticipantMapping | None) -> dict[str, tuple[tuple[str, ...], str | None]]:
    if document is None or mapping is None:
        return {}
    project_idx = document.column(mapping.project_id_column)
    organisation_idx = document.column(mapping.organisation_id_column)
    role_idx = document.column(mapping.role_column) if mapping.role_column else None
    members: dict[str, set[str]] = {}
    coordinators: dict[str, str] = {}
    for row in document.rows:
        project, organisation = row.cells[project_idx], row.cells[organisation_idx]
        if not project or not organisation or len(project) > 200 or len(organisation) > 200:
            raise TaskFailure(f"Участник CORDIS в строке {row.line} не имеет ID.")
        members.setdefault(project, set()).add(organisation)
        if len(members[project]) > 200:
            raise TaskFailure("В проекте CORDIS слишком много участников для одного импорта.")
        if role_idx is not None and row.cells[role_idx].casefold() == "coordinator":
            previous = coordinators.setdefault(project, organisation)
            if previous != organisation:
                raise TaskFailure("У проекта CORDIS несколько разных координаторов.")
    return {project: (tuple(sorted(organisations)), coordinators.get(project))
            for project, organisations in members.items()}


def parse_cordis_projects(document: CsvDocument, mapping: CordisMapping, *, raw_hash: str,
                          observed_at: datetime, participants: CsvDocument | None = None,
                          participant_mapping: ParticipantMapping | None = None) -> CapitalParse:
    if (participants is None) != (participant_mapping is None):
        raise TaskFailure("Для участников CORDIS нужна отдельная карта колонок.")
    member_index = _participants(participants, participant_mapping)
    columns = (mapping.project_id_column, mapping.title_column, mapping.objective_column,
               mapping.ec_contribution_column, mapping.ec_signature_column)
    indexes = tuple(document.column(name) for name in columns)
    if len(set(indexes)) != len(indexes):
        raise TaskFailure("Поля CORDIS должны указывать на разные колонки.")
    events = []
    descriptions = []
    rejected = []
    seen: set[str] = set()
    for row in document.rows:
        try:
            project_id, title, objective, amount_raw, signed_raw = (row.cells[index] for index in indexes)
            if not project_id or len(project_id) > 200 or not title or not objective:
                raise ValueError("project_identity_or_objective")
            if project_id in seen:
                raise TaskFailure("Один project ID CORDIS повторяется в одной выгрузке; требуется разрешить конфликт.")
            seen.add(project_id)
            amount = _money(amount_raw, mapping.money_format)
            signature, _, _ = _date(signed_raw, mapping.signature_format)
            if signature is not None and signature > observed_at.date():
                raise ValueError("future_signature")
            beneficiaries, participant_coordinator = member_index.get(project_id, ((), None))
            coordinator = _cell(document, row.cells, mapping.coordinator_column) or participant_coordinator
            if coordinator and participant_coordinator and coordinator != participant_coordinator:
                raise ValueError("coordinator_conflict")
            identifier = uuid5(NAMESPACE_URL, "cordis:project:" + project_id)
            event = CapitalEvent(event_id=identifier, source_event_id=project_id,
                                 source="cordis", source_hash=raw_hash, kind="grant_project",
                                 status="confirmed" if signature else "unknown", project_id=project_id,
                                 beneficiary_ids=beneficiaries, coordinator_id=coordinator,
                                 agreement_at=signature, event_date_precision="day" if signature else "unknown",
                                 observed_at=observed_at, available_at=observed_at, amount=amount,
                                 currency="EUR", amount_status="undisclosed" if amount is None else
                                 "raw_zero_unverified" if Decimal(amount) == 0 else "disclosed",
                                 amount_kind="eu_contribution", programme=_cell(document, row.cells,
                                                                                mapping.programme_column),
                                 export_right="local_only")
            description = CapitalDescription(source="cordis", source_event_id=project_id, event_id=identifier,
                                             source_hash=raw_hash, title=title, description=objective,
                                             source_url="https://cordis.europa.eu/project/id/" + project_id
                                             if re.fullmatch(r"[0-9]{1,20}", project_id) else None)
            events.append(event)
            descriptions.append(description)
        except TaskFailure:
            raise
        except (ValueError, OverflowError) as error:
            rejected.append(f"line:{row.line}:{str(error)[:80]}")
    return CapitalParse(tuple(events), tuple(descriptions), tuple(rejected))


def parse_investments(document: CsvDocument, mapping: InvestmentMapping, *, raw_hash: str,
                      observed_at: datetime) -> CapitalParse:
    required = (mapping.event_id_column, mapping.recipient_id_column, mapping.recipient_name_column,
                mapping.kind_column, mapping.status_column, mapping.date_column, mapping.amount_column,
                mapping.currency_column)
    indexes = tuple(document.column(name) for name in required)
    if len(set(indexes)) != len(indexes):
        raise TaskFailure("Поля инвестиций должны указывать на разные колонки.")
    events = []
    descriptions = []
    rejected = []
    seen: set[str] = set()
    for row in document.rows:
        try:
            source_id, recipient_id, name, kind, status, date_raw, amount_raw, currency = (
                row.cells[index] for index in indexes)
            if not source_id or not recipient_id or not name or len(source_id) > 200:
                raise ValueError("investment_identity")
            if source_id in seen:
                raise TaskFailure("Один ID инвестиционного события повторяется в одной выгрузке.")
            seen.add(source_id)
            if kind not in {"equity_round", "debt", "acquisition"} or status not in {
                    "announced", "confirmed", "cancelled", "rumored", "unknown"}:
                raise ValueError("investment_kind_or_status")
            typed_kind = cast(Literal["equity_round", "debt", "acquisition"], kind)
            typed_status = cast(Literal["announced", "confirmed", "cancelled", "rumored", "unknown"], status)
            amount = _money(amount_raw, mapping.money_format)
            precise, month, year = _date(date_raw, mapping.date_format)
            if (precise is not None and precise > observed_at.date()
                    or month is not None and month > observed_at.strftime("%Y-%m")
                    or year is not None and year > observed_at.year):
                raise ValueError("future_event")
            if amount is not None and (not re.fullmatch(r"[A-Z]{3}", currency)):
                raise ValueError("currency")
            identifier = uuid5(NAMESPACE_URL, "investment:" + source_id)
            amount_kind = cast(Literal["round_amount", "debt_amount", "purchase_price"], {
                "equity_round": "round_amount", "debt": "debt_amount", "acquisition": "purchase_price"}[typed_kind])
            event = CapitalEvent(event_id=identifier, source_event_id=source_id,
                                 source="investment_csv", source_hash=raw_hash, kind=typed_kind,
                                 status=typed_status, recipient_id=recipient_id, recipient_name=name,
                                 announced_at=precise, event_month=month, event_year=year,
                                 event_date_precision="day" if precise else "month" if month else "year" if year else "unknown",
                                 observed_at=observed_at, available_at=observed_at,
                                 amount=amount, currency=currency or None,
                                 amount_status="undisclosed" if amount is None else
                                 "raw_zero_unverified" if Decimal(amount) == 0 else "disclosed",
                                 amount_kind=amount_kind, export_right="local_only")
            description = CapitalDescription(source="investment_csv", source_event_id=source_id,
                                             event_id=identifier, source_hash=raw_hash, title=name,
                                             description=_cell(document, row.cells, mapping.description_column),
                                             source_url=_cell(document, row.cells, mapping.url_column))
            events.append(event)
            descriptions.append(description)
        except TaskFailure:
            raise
        except (ValueError, OverflowError) as error:
            rejected.append(f"line:{row.line}:{str(error)[:80]}")
    return CapitalParse(tuple(events), tuple(descriptions), tuple(rejected))


def import_capital_csv(store: SignalStore, path: Path, profile_hash: str, *,
                       mapping: CordisMapping | InvestmentMapping, encoding: CsvEncoding,
                       delimiter: CsvDelimiter, retention: str, observed_at: datetime | None = None,
                       participant_path: Path | None = None, participant_mapping: ParticipantMapping | None = None,
                       cancel: Event | None = None, export_right: ExportRight = "local_only",
                       license_ref: str | None = None) -> str:
    validate_import_export_right(export_right, license_ref)
    store.get_object(profile_hash, QueryProfile)
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise TaskFailure("Дата импорта должна включать часовой пояс.")
    if isinstance(mapping, InvestmentMapping) and (participant_path is not None or participant_mapping is not None):
        raise TaskFailure("Таблица участников относится только к грантовым проектам.")
    raw_hash = store.put_raw(path, "csv", retention=retention, cancel=cancel)
    document = read_csv_document(store.verify_raw(raw_hash, "csv", cancel=cancel), encoding=encoding,
                                 delimiter=delimiter, cancel=cancel)
    participant_hash = None
    participants = None
    if participant_path is not None:
        participant_hash = store.put_raw(participant_path, "csv", retention=retention, cancel=cancel)
        participants = read_csv_document(store.verify_raw(participant_hash, "csv", cancel=cancel),
                                         encoding=encoding, delimiter=delimiter, cancel=cancel)
    source: Literal["cordis", "investment_csv"] = "cordis" if isinstance(mapping, CordisMapping) else "investment_csv"
    if isinstance(mapping, CordisMapping):
        parsed = parse_cordis_projects(document, mapping, raw_hash=raw_hash, observed_at=timestamp,
                                       participants=participants, participant_mapping=participant_mapping)
    else:
        parsed = parse_investments(document, mapping, raw_hash=raw_hash, observed_at=timestamp)
    mapping_hash = content_hash({"mapping": asdict(mapping), "participants": asdict(participant_mapping)
                                 if participant_mapping is not None else None,
                                 "encoding": encoding, "delimiter": delimiter})
    snapshot = SourceSnapshot(source=source, adapter_version=source + "-csv/1",
                              request_hash=content_hash({"source": source, "mapping_hash": mapping_hash,
                                                         "profile": profile_hash}),
                              query_profile_hash=profile_hash, observed_at=timestamp, available_at=timestamp,
                              coverage="partial", comparable=False, raw_hash=raw_hash,
                              retention="local_allowed", export_right=export_right, license_ref=license_ref,
                              limitations=("Импортированная выгрузка не доказывает полного покрытия рынка.",))
    snapshot_hash = store.put_object(snapshot, cancel=cancel)
    event_hashes = tuple(store.put_object(CapitalEvent.model_validate(item.model_copy(update={
        "export_right": export_right, "license_ref": license_ref,
    }).model_dump(mode="json")), cancel=cancel) for item in parsed.events)
    description_hashes = tuple(store.put_object(item, cancel=cancel) for item in parsed.descriptions)
    store.verify_raw(raw_hash, "csv", cancel=cancel)
    if participant_hash is not None:
        store.verify_raw(participant_hash, "csv", cancel=cancel)
    receipt = CapitalImportReceipt(source=source, query_profile_hash=profile_hash, snapshot_hash=snapshot_hash,
                                   raw_hash=raw_hash, participant_raw_hash=participant_hash, mapping_hash=mapping_hash,
                                   row_count=len(document.rows), rejected_count=len(parsed.rejected),
                                   event_hashes=event_hashes, description_hashes=description_hashes,
                                   rejected_rows=parsed.rejected, completed_at=timestamp)
    return store.put_object(receipt, cancel=cancel)


def propose_capital_association(concept: TechnologyConcept, event: CapitalEvent,
                                description: CapitalDescription, description_hash: str) -> TechnologyAssociation | None:
    """Exact lexical evidence can suggest a link but never confirm money attribution."""
    if (description.event_id != event.event_id or description.source_hash != event.source_hash
            or description.source != event.source or object_digest(description) != description_hash):
        raise TaskFailure("Описание финансового события не соответствует исходной записи.")
    text = (description.title + " " + (description.description or "")).casefold()
    names = (concept.label, *(term.text for term in concept.aliases if term.status == "confirmed"))
    if not any(re.search(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", text) for name in names):
        return None
    return TechnologyAssociation(concept_id=concept.concept_id,
                                 subject_id=(event.project_id or "") if event.kind == "grant_project" else
                                 (event.recipient_id or ""),
                                 subject_kind="project" if event.kind == "grant_project" else "organisation",
                                 relation="researches" if event.kind == "grant_project" else "mentioned",
                                 status="proposed", evidence_hashes=(description_hash,),
                                 relation_at_event="unknown")

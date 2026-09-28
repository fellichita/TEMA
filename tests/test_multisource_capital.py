"""One project is one grant; unreviewed company links carry no tech money."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.pilot.multisource.capital import (CordisMapping, InvestmentMapping, ParticipantMapping,
                                           import_capital_csv, parse_cordis_projects, parse_investments,
                                           propose_capital_association)
from app.pilot.multisource.contracts import (CapitalDescription, CapitalEvent, CapitalImportReceipt,
                                             SourceSnapshot, TechnologyConcept)
from app.pilot.multisource.imports import read_csv_document
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore, object_digest
from app.runtime.jobs import TaskFailure


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
HASH = "a" * 64


def _csv(tmp_path: Path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path, read_csv_document(path, encoding="utf-8-sig", delimiter=";")


def _cordis_mapping():
    return CordisMapping(project_id_column="id", title_column="title", objective_column="objective",
                         ec_contribution_column="ecMaxContribution", ec_signature_column="ecSignatureDate",
                         money_format="decimal_dot", signature_format="YYYY-MM-DD")


def _investment_mapping(date_format="YYYY-MM"):
    return InvestmentMapping(event_id_column="id", recipient_id_column="company_id",
                             recipient_name_column="company", kind_column="kind", status_column="status",
                             date_column="date", date_format=date_format, amount_column="amount",
                             currency_column="currency", money_format="decimal_dot",
                             description_column="description")


def _profile():
    return build_manual_profile("молекулярная память", "Запись цифровых данных в ДНК",
                                seed_terms=("ДНК память",), primary_phrase="ДНК память", confirmed_at=NOW)


def test_cordis_grant_amount_is_not_multiplied_by_ten_participants(tmp_path: Path) -> None:
    _, project = _csv(tmp_path, "project.csv", "id;title;objective;ecMaxContribution;totalCost;ecSignatureDate\n"
                      "101234567;ДНК память;Технология хранения данных;1000000.00;2500000.00;2026-07-09\n")
    participant_rows = "project;organisation;role\n" + "".join(
        f"101234567;ORG{index};{'coordinator' if index == 0 else 'participant'}\n" for index in range(10))
    _, participants = _csv(tmp_path, "participants.csv", participant_rows)
    parsed = parse_cordis_projects(project, _cordis_mapping(), raw_hash=HASH, observed_at=NOW,
                                   participants=participants,
                                   participant_mapping=ParticipantMapping("project", "organisation", "role"))
    assert parsed.rejected == () and len(parsed.events) == 1
    event = parsed.events[0]
    assert len(event.beneficiary_ids) == 10 and event.coordinator_id == "ORG0"
    assert event.amount == "1000000.00" and event.amount_kind == "eu_contribution"
    assert event.agreement_at == date(2026, 7, 9) and event.status == "confirmed"
    assert parsed.descriptions[0].source_url == "https://cordis.europa.eu/project/id/101234567"


def test_unknown_signature_and_raw_zero_do_not_become_positive_grant(tmp_path: Path) -> None:
    _, project = _csv(tmp_path, "project.csv", "id;title;objective;ecMaxContribution;ecSignatureDate\n"
                      "1;ДНК память;Исследование;0;\n")
    event = parse_cordis_projects(project, _cordis_mapping(), raw_hash=HASH, observed_at=NOW).events[0]
    assert event.amount_status == "raw_zero_unverified" and event.agreement_at is None
    assert event.event_date_precision == "unknown" and event.status == "unknown"


def test_investment_month_keeps_month_precision_and_unknown_amount(tmp_path: Path) -> None:
    _, document = _csv(tmp_path, "rounds.csv", "id;company_id;company;kind;status;date;amount;currency;description\n"
                       "round-1;firm-1;Example;equity_round;confirmed;2026-08;;USD;ДНК память\n")
    parsed = parse_investments(document, _investment_mapping(), raw_hash=HASH, observed_at=NOW)
    event = parsed.events[0]
    assert event.event_month == "2026-08" and event.announced_at is None
    assert event.event_date_precision == "month" and event.amount is None
    assert event.amount_status == "undisclosed" and event.currency == "USD"
    future = _csv(tmp_path, "future.csv", "id;company_id;company;kind;status;date;amount;currency;description\n"
                  "round-2;firm-2;Example;equity_round;confirmed;2027-01;100;USD;ДНК память\n")[1]
    assert len(parse_investments(future, _investment_mapping(), raw_hash=HASH, observed_at=NOW).rejected) == 1


def test_company_mention_proposes_only_unattributed_link(tmp_path: Path) -> None:
    _, document = _csv(tmp_path, "rounds.csv", "id;company_id;company;kind;status;date;amount;currency;description\n"
                       "round-1;firm-1;Example;equity_round;confirmed;2026-08-15;2000000;USD;ДНК память\n")
    parsed = parse_investments(document, _investment_mapping("YYYY-MM-DD"), raw_hash=HASH, observed_at=NOW)
    concept = TechnologyConcept(concept_id=_profile().profile_id, label="ДНК память",
                                definition="Молекулярная память", identity_status="confirmed", confirmed_at=NOW,
                                provenance_hashes=(HASH,))
    description = parsed.descriptions[0]
    association = propose_capital_association(concept, parsed.events[0], description, object_digest(description))
    assert association is not None and association.status == "proposed" and association.relation == "mentioned"
    assert association.attributable_amount is None and association.relation_at_event == "unknown"


def test_duplicate_ids_fail_without_double_count(tmp_path: Path) -> None:
    _, project = _csv(tmp_path, "project.csv", "id;title;objective;ecMaxContribution;ecSignatureDate\n"
                      "1;A;Research;100;2026-01-01\n1;A;Research;100;2026-01-01\n")
    with pytest.raises(TaskFailure, match="повторяется"):
        parse_cordis_projects(project, _cordis_mapping(), raw_hash=HASH, observed_at=NOW)


def test_capital_import_receipt_closes_over_project_and_participant_raw(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path, _ = _csv(tmp_path, "project.csv", "id;title;objective;ecMaxContribution;ecSignatureDate\n"
                   "101234567;ДНК память;Молекулярный архив;1000000;2026-07-09\n")
    participants, _ = _csv(tmp_path, "participants.csv", "project;organisation;role\n"
                           "101234567;ORG1;coordinator\n101234567;ORG2;participant\n")
    receipt_hash = import_capital_csv(store, path, profile_hash, mapping=_cordis_mapping(),
                                      encoding="utf-8-sig", delimiter=";", retention="local_allowed",
                                      observed_at=NOW, participant_path=participants,
                                      participant_mapping=ParticipantMapping("project", "organisation", "role"))
    receipt = store.get_object(receipt_hash, CapitalImportReceipt)
    assert receipt.source == "cordis" and receipt.participant_raw_hash is not None
    assert receipt.row_count == 1 and len(receipt.event_hashes) == 1
    assert store.get_object(receipt.snapshot_hash, SourceSnapshot).comparable is False
    assert store.get_object(receipt.event_hashes[0], CapitalEvent).amount == "1000000"
    assert store.get_object(receipt.description_hashes[0], CapitalDescription).title == "ДНК память"
    assert store.verify_raw(receipt.participant_raw_hash, "csv").exists()
    with pytest.raises(TaskFailure):
        import_capital_csv(store, path, profile_hash, mapping=_cordis_mapping(),
                           encoding="utf-8-sig", delimiter=";", retention="unknown", observed_at=NOW)


def test_capital_transfer_rights_cover_snapshot_and_events(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    profile_hash = store.put_object(_profile())
    path, _ = _csv(tmp_path, "project.csv", "id;title;objective;ecMaxContribution;ecSignatureDate\n"
                   "101234567;ДНК память;Молекулярный архив;1000000;2026-07-09\n")
    receipt_hash = import_capital_csv(store, path, profile_hash, mapping=_cordis_mapping(),
                                      encoding="utf-8-sig", delimiter=";", retention="local_allowed",
                                      observed_at=NOW, export_right="share_allowed",
                                      license_ref="public project export")
    receipt = store.get_object(receipt_hash, CapitalImportReceipt)
    assert store.get_object(receipt.snapshot_hash, SourceSnapshot).export_right == "share_allowed"
    event = store.get_object(receipt.event_hashes[0], CapitalEvent)
    assert event.export_right == "share_allowed" and event.license_ref == "public project export"

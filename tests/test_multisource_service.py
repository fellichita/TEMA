"""Real coordinator publication, restart, and source-closure checks."""

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.pilot.multisource.capital import CordisMapping, import_capital_csv
from app.pilot.export import export_result
from app.pilot.multisource.contracts import CapitalImportReceipt, TechnologyConcept
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore
from app.pilot.multisource.wordstat import DynamicsMapping, import_wordstat_csv
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure
from tests.test_pilot_export import make_result
from tests.test_multisource_arxiv import SCOPE, _entry, _feed


def _service(data_dir: Path) -> PilotService:
    return PilotService(data_dir, CredentialStore())


def _profile(store: SignalStore):
    when = datetime.now(timezone.utc)
    query = build_manual_profile("Молекулярная память", "Хранение цифровых данных в молекулах",
                                 seed_terms=("ДНК память",), primary_phrase="ДНК память", confirmed_at=when)
    query_hash = store.put_object(query)
    concept = TechnologyConcept(concept_id=uuid4(), label="ДНК память", definition=query.definition,
                                identity_status="confirmed", confirmed_at=when,
                                provenance_hashes=(query_hash,))
    return query_hash, store.put_object(concept), concept


def _wordstat(tmp_path: Path, store: SignalStore, query_hash: str) -> str:
    rows = ["Месяц;Запросов;Доля"]
    for year in (2025, 2026):
        for month in range(1, 13):
            if year == 2026 and month > 8:
                break
            recent = year == 2026 and month in (6, 7, 8)
            count = 60 if recent else 30
            share = "0,20%" if recent else "0,10%"
            rows.append(f"{month:02d}.{year};{count};{share}")
    source = tmp_path / "wordstat.csv"
    source.write_bytes(("\n".join(rows) + "\n").encode("utf-8-sig"))
    mapping = DynamicsMapping(date_column="Месяц", count_column="Запросов", share_column="Доля",
                              date_format="MM.YYYY", share_unit="percent", phrase="ДНК память",
                              expected_from=date(2025, 1, 1), expected_to=date(2026, 8, 1))
    return import_wordstat_csv(store, source, query_hash, kind="dynamics", mapping=mapping,
                               encoding="utf-8-sig", delimiter=";", retention="local_allowed")


def test_wordstat_profile_publishes_once_and_reopens_without_scientific_history(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    store = SignalStore(data_dir)
    query_hash, concept_hash, _ = _profile(store)
    receipt_hash = _wordstat(tmp_path, store, query_hash)
    service = _service(data_dir)
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt_hash)
        service.coordinator.wait()
        assert service.get(run)["state"] == "succeeded"
        profile = service.signal_result(run)["profile"]
        assert len(profile["attention_ids"]) == 1
        assert profile["findings"][0]["search_state"] == "sustained_growth"
        evidence = service.signal_finding_evidence(run, profile["findings"][0]["finding_id"])
        assert evidence["metrics"][0]["metric_kind"] == "SearchMetric"
        assert evidence["total_observations"] >= 6
        assert all(item["kind"] == "SearchObservation" for item in evidence["observations"])
        scenario = service.signal_scenario(run, "exclude_wordstat")
        assert scenario["attention_before"] == 1 and scenario["attention_after"] == 0
        assert scenario["changes"][0]["before_queue"] == "attention"
        assert scenario["changes"][0]["after_queue"] == "insufficient_data"
        assert scenario == service.signal_scenario(run, "exclude_wordstat")
        assert service.signal_result(run)["profile"] == profile
        later = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt_hash)
        service.coordinator.wait()
        comparison = service.signal_compare(run, later)
        assert comparison["comparable"] is True
        assert comparison["changed_findings"] == []
        assert comparison["event_change_count"] == 0
        assert comparison == service.signal_compare(run, later)
        assert service.list_runs(source="local") == []
        assert {item["id"] for item in service.list_signal_runs()} == {run, later}
        with pytest.raises(TaskFailure):
            service.result(run)
    finally:
        service.close()
    reopened = _service(data_dir)
    try:
        assert reopened.signal_result(run)["profile"] == profile
    finally:
        reopened.close()


def test_proposed_grant_is_watch_until_reviewed_and_source_corruption_blocks_read(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    store = SignalStore(data_dir)
    query_hash, concept_hash, _ = _profile(store)
    source = tmp_path / "cordis.csv"
    source.write_text("id;title;objective;ecMaxContribution;ecSignatureDate\n"
                      "101;ДНК память;Исследование молекулярной памяти;1000000;2026-08-15\n",
                      encoding="utf-8")
    mapping = CordisMapping(project_id_column="id", title_column="title", objective_column="objective",
                            ec_contribution_column="ecMaxContribution", ec_signature_column="ecSignatureDate",
                            money_format="decimal_dot", signature_format="YYYY-MM-DD")
    receipt_hash = import_capital_csv(store, source, query_hash, mapping=mapping,
                                      encoding="utf-8-sig", delimiter=";", retention="local_allowed")
    service = _service(data_dir)
    try:
        first = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(receipt_hash,))
        service.coordinator.wait()
        assert service.get(first)["state"] == "succeeded"
        initial = service.signal_result(first)["profile"]
        assert initial["attention_ids"] == [] and len(initial["watch_ids"]) == 1
        receipt = store.get_object(receipt_hash, CapitalImportReceipt)
        proposals = service.signal_associations(first)
        assert len(proposals) == 1 and "ДНК память" in proposals[0]["title"]
        association_hash = service.confirm_signal_grant(first, proposals[0]["hash"], "local-analyst")
        second = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(receipt_hash,),
                                       association_hashes=(association_hash,))
        service.coordinator.wait()
        assert service.get(second)["state"] == "succeeded", service.get(second)["error"]
        confirmed = service.signal_result(second)["profile"]
        assert len(confirmed["attention_ids"]) == 1
        assert confirmed["findings"][0]["funding_state"] == "single_unit"
        assert service.signal_result(first)["profile"] == initial
        raw = store.verify_raw(receipt.raw_hash, "csv")
        raw.write_bytes(b"tampered")
        with pytest.raises(TaskFailure):
            service.signal_result(second)
    finally:
        service.close()


def test_verified_existing_science_card_is_linked_without_rerunning_ml(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    result, archive, artifacts = make_result(data_dir, historical=True)
    package = export_result(tmp_path / "science.trendresult", result, archive, artifacts)
    store = SignalStore(data_dir)
    when = datetime.now(timezone.utc)
    label = result.cards[0].candidate.label
    query = build_manual_profile(label, result.cards[0].candidate.definition,
                                 seed_terms=(label,), confirmed_at=when)
    query_hash = store.put_object(query)
    concept = TechnologyConcept(concept_id=uuid4(), label=label, definition=query.definition,
                                identity_status="confirmed", confirmed_at=when,
                                provenance_hashes=(query_hash,))
    concept_hash = store.put_object(concept)
    service = _service(data_dir)
    try:
        imported = service.import_result(str(package.path))["id"]
        assert any(item["run_id"] == imported for item in service.signal_scientific_runs())
        assert service.signal_scientific_cards(imported)[0]["candidate_id"] == result.cards[0].candidate.candidate_id
        run = service.start_signals(query_hash, (concept_hash,), base_result_run_id=imported,
                                    scientific_links=({"concept_id": str(concept.concept_id),
                                                       "candidate_id": result.cards[0].candidate.candidate_id},))
        service.coordinator.wait()
        assert service.get(run)["state"] == "succeeded", service.get(run)["error"]
        profile = service.signal_result(run)["profile"]
        assert profile["findings"][0]["scientific_category"] == result.cards[0].category
        assert profile["base_result_run_id"] == imported
        assert service.result(imported)["result"] == result.model_dump(mode="json")
    finally:
        service.close()


def test_cancelled_profile_keeps_checkpoints_but_only_resume_publishes(tmp_path: Path, monkeypatch) -> None:
    from app.pilot.multisource import workflow

    data_dir = tmp_path / "app"
    store = SignalStore(data_dir)
    query_hash, concept_hash, _ = _profile(store)
    receipt_hash = _wordstat(tmp_path, store, query_hash)
    stage = workflow._stage
    interrupt = True

    def cancel_after_metrics(context, name, value, message, number):
        nonlocal interrupt
        stage(context, name, value, message, number)
        if name == "signals_metrics" and interrupt:
            interrupt = False
            context.cancel_event.set()

    monkeypatch.setattr(workflow, "_stage", cancel_after_metrics)
    service = _service(data_dir)
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt_hash)
        service.coordinator.wait()
        assert service.get(run)["state"] == "cancelled"
        assert service.coordinator.checkpoint_value(run, "signals_metrics") is not None
        with pytest.raises(TaskFailure):
            service.signal_result(run)
        service.resume(run)
        service.coordinator.wait()
        assert service.get(run)["state"] == "succeeded", service.get(run)["error"]
        assert service.signal_result(run)["profile"]["attention_ids"]
    finally:
        service.close()


def test_desktop_facade_creates_previews_imports_and_runs_without_a_key(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    service = _service(data_dir)
    try:
        created = service.create_signal_query("Молекулярная память", "Хранение данных в молекулах", "ДНК память")
        store = SignalStore(data_dir)
        _wordstat(tmp_path, store, created["query_profile_hash"])
        source = tmp_path / "wordstat.csv"
        assert service.preview_signal_csv(str(source), "utf-8-sig", ";")["row_count"] == 20
        mapping = dict(date_column="Месяц", count_column="Запросов", share_column="Доля",
                       date_format="MM.YYYY", share_unit="percent", phrase="ДНК память",
                       expected_from="2025-01", expected_to="2026-08")
        with pytest.raises(TaskFailure):
            service.import_signal_csv(created["query_profile_hash"], str(source), "wordstat", mapping,
                                      "utf-8-sig", ";", retention_confirmed=False)
        imported = service.import_signal_csv(created["query_profile_hash"], str(source), "wordstat", mapping,
                                             "utf-8-sig", ";", retention_confirmed=True)
        run = service.start_signals(created["query_profile_hash"], (created["concept_hash"],),
                                    wordstat_receipt_hash=imported["receipt_hash"])
        service.coordinator.wait()
        displayed = service.signal_result(run)
        assert displayed["profile"]["attention_ids"]
        assert displayed["concepts"][created["concept_id"]] == "ДНК память"
        assert displayed["imports"] == {"wordstat": imported["receipt_hash"]}
    finally:
        service.close()


def test_explicit_arxiv_link_is_watch_and_unlinked_preprint_is_not_claimed(tmp_path: Path) -> None:
    service = _service(tmp_path / "app")
    try:
        created = service.create_signal_query(SCOPE, "Selective transport through a membrane", SCOPE)
        path = _feed(tmp_path / "export.atom", _entry(1))
        imported = service.import_signal_atom(created["query_profile_hash"], str(path), retention_confirmed=True)
        candidates = service.signal_arxiv_candidates(imported["receipt_hash"])
        assert len(candidates) == 1 and "membrane" in candidates[0]["title"].lower()
        first = service.start_signals(created["query_profile_hash"], (created["concept_hash"],),
                                      arxiv_receipt_hash=imported["receipt_hash"])
        service.coordinator.wait()
        unlinked = service.signal_result(first)["profile"]
        assert unlinked["attention_ids"] == []
        assert unlinked["findings"][0]["rule_id"] == "insufficient_channels"
        linked_hash = service.link_signal_arxiv(created["concept_hash"], imported["receipt_hash"],
                                                candidates[0]["revision_id"])
        second = service.start_signals(created["query_profile_hash"], (linked_hash,),
                                       arxiv_receipt_hash=imported["receipt_hash"])
        service.coordinator.wait()
        linked = service.signal_result(second)["profile"]
        assert linked["attention_ids"] == [] and len(linked["watch_ids"]) == 1
        assert linked["findings"][0]["rule_id"] == "arxiv_unassessed"
        assert "arxiv" in linked["findings"][0]["origins"]
    finally:
        service.close()

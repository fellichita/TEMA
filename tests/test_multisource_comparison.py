"""Saved-run comparisons distinguish corrected facts from changed settings."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.pilot.multisource.capital import CordisMapping, import_capital_csv
from app.pilot.multisource.contracts import SignalProfile, TechnologyAssociation
from app.pilot.multisource.profiles import compare_profiles
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from tests.test_multisource_service import _profile, _wordstat


def _grant(tmp_path: Path, store: SignalStore, query_hash: str, amount: int) -> str:
    path = tmp_path / "cordis.csv"
    path.write_text("id;title;objective;ecMaxContribution;ecSignatureDate\n"
                    f"101;ДНК память;Исследование молекулярной памяти;{amount};2024-08-15\n",
                    encoding="utf-8")
    mapping = CordisMapping(project_id_column="id", title_column="title", objective_column="objective",
                            ec_contribution_column="ecMaxContribution", ec_signature_column="ecSignatureDate",
                            money_format="decimal_dot", signature_format="YYYY-MM-DD")
    return import_capital_csv(store, path, query_hash, mapping=mapping, encoding="utf-8-sig",
                              delimiter=";", retention="local_allowed")


def test_old_grant_first_imported_then_corrected_without_double_counting(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, _ = _profile(store)
    search = _wordstat(tmp_path, store, query_hash)
    service = PilotService(tmp_path, CredentialStore())
    try:
        before = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=search)
        service.coordinator.wait()
        first_grant = _grant(tmp_path, store, query_hash, 1000000)
        middle = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=search,
                                       capital_receipt_hashes=(first_grant,))
        service.coordinator.wait()
        first_change = service.signal_compare(before, middle)
        assert first_change["comparable"] is True
        assert first_change["event_change_count"] == 1
        assert first_change["event_changes"][0]["kind"] == "older_event_first_imported"
        corrected = _grant(tmp_path, store, query_hash, 1200000)
        last = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=search,
                                     capital_receipt_hashes=(corrected,))
        service.coordinator.wait()
        second_change = service.signal_compare(middle, last)
        assert second_change["event_change_count"] == 1
        assert second_change["event_changes"][0]["kind"] == "corrected_event"
    finally:
        service.close()


def test_changed_aliases_block_priority_comparison(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, _ = _profile(store)
    search = _wordstat(tmp_path, store, query_hash)
    service = PilotService(tmp_path, CredentialStore())
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=search)
        service.coordinator.wait()
        baseline = SignalProfile.model_validate(service.signal_result(run)["profile"])
        revised_query = build_manual_profile("Молекулярная память", "Хранение цифровых данных в молекулах",
            seed_terms=("ДНК память", "DNA storage"), primary_phrase="ДНК память",
            confirmed_at=datetime.now(timezone.utc))
        revised_hash = store.put_object(revised_query)
        revised = SignalProfile.model_validate(baseline.model_copy(update={
            "query_profile_hash": revised_hash, "decision_at": baseline.decision_at + timedelta(seconds=1),
            "knowledge_cutoff": baseline.knowledge_cutoff + timedelta(seconds=1),
            "collection_finished_at": baseline.collection_finished_at + timedelta(seconds=1),
        }).model_dump(mode="json"))
        comparison = compare_profiles(store, baseline, revised)
        assert comparison["comparable"] is False
        assert "different_query_semantics" in comparison["reasons"]
        assert comparison["changed_findings"] == []
    finally:
        service.close()


def test_reviewed_and_revoked_relation_are_explicit_comparison_reasons(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, _ = _profile(store)
    grant = _grant(tmp_path, store, query_hash, 1000000)
    service = PilotService(tmp_path, CredentialStore())
    try:
        proposed = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(grant,))
        service.coordinator.wait()
        proposal_hash = service.signal_associations(proposed)[0]["hash"]
        confirmed_hash = service.confirm_signal_grant(proposed, proposal_hash, "reviewer")
        confirmed = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(grant,),
                                          association_hashes=(confirmed_hash,))
        service.coordinator.wait()
        first_change = service.signal_compare(proposed, confirmed)
        assert first_change["association_change_count"] == 1
        assert first_change["association_changes"][0]["kind"] == "new_confirmed_relation"
        previous = store.get_object(confirmed_hash, TechnologyAssociation)
        rejected = previous.model_copy(update={"status": "rejected", "relation_at_event": "not_applicable"})
        rejected_hash = store.put_object(TechnologyAssociation.model_validate(rejected.model_dump(mode="json")))
        revoked = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(grant,),
                                        association_hashes=(rejected_hash,))
        service.coordinator.wait()
        second_change = service.signal_compare(confirmed, revoked)
        assert second_change["association_changes"][0]["kind"] == "revoked_relation"
    finally:
        service.close()

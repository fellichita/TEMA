"""Ablation of one disclosed event is an offline, reversible view."""

from pathlib import Path

import pytest

from app.pilot.multisource.capital import CordisMapping, import_capital_csv
from app.pilot.multisource.store import SignalStore
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure
from tests.test_multisource_service import _profile


@pytest.mark.parametrize("second_amount, expected_options", [(1000000, 1), (2000000, 2)])
def test_largest_grant_recalculates_project_breadth_without_mutating_profile(
        tmp_path: Path, second_amount: int, expected_options: int) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, concept = _profile(store)
    source = tmp_path / "grants.csv"
    source.write_text(
        "id;title;objective;ecMaxContribution;ecSignatureDate\n"
        "101;ДНК память;Исследование молекулярной памяти;2000000;2026-08-15\n"
        f"102;ДНК память;Исследование молекулярной памяти;{second_amount};2026-08-16\n",
        encoding="utf-8",
    )
    mapping = CordisMapping(project_id_column="id", title_column="title", objective_column="objective",
                            ec_contribution_column="ecMaxContribution", ec_signature_column="ecSignatureDate",
                            money_format="decimal_dot", signature_format="YYYY-MM-DD")
    receipt = import_capital_csv(store, source, query_hash, mapping=mapping,
                                 encoding="utf-8-sig", delimiter=";", retention="local_allowed")
    service = PilotService(tmp_path, CredentialStore())
    try:
        proposed_run = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(receipt,))
        service.coordinator.wait()
        proposal_hashes = [item["hash"] for item in service.signal_associations(proposed_run)]
        assert len(proposal_hashes) == 2
        reviewed = tuple(service.confirm_signal_grant(proposed_run, digest, "reviewer")
                         for digest in proposal_hashes)
        run = service.start_signals(query_hash, (concept_hash,), capital_receipt_hashes=(receipt,),
                                    association_hashes=reviewed)
        service.coordinator.wait()
        baseline = service.signal_result(run)["profile"]
        assert baseline["findings"][0]["funding_state"] == "multiple_units"
        options = service.signal_largest_event_options(run, str(concept.concept_id))
        assert len(options) == expected_options and all(item["amount"] == "2000000" for item in options)
        option = options[0]
        scenario = service.signal_scenario(run, "exclude_largest_disclosed_event", str(concept.concept_id),
                                           "grant_project", "EUR", option["event_hash"])
        assert scenario["excluded_event"]["event_hash"] == option["event_hash"]
        assert scenario["changes"][0]["before_queue"] == "attention"
        assert scenario["changes"][0]["after_queue"] == "attention"
        assert scenario["changes"][0]["before_funding"] == "multiple_units"
        assert scenario["changes"][0]["after_funding"] == "single_unit"
        assert scenario["changes"][0]["funding_units"] == (2, 1)
        assert scenario["scenario_profile_hash"] != scenario["baseline_profile_hash"]
        assert service.signal_result(run)["profile"] == baseline
        with pytest.raises(TaskFailure, match="валюты"):
            service.signal_scenario(run, "exclude_largest_disclosed_event", str(concept.concept_id),
                                    "grant_project", "USD")
        with pytest.raises(TaskFailure, match="крупнейших"):
            service.signal_scenario(run, "exclude_largest_disclosed_event", str(concept.concept_id),
                                    "grant_project", "EUR", "f" * 64)
    finally:
        service.close()

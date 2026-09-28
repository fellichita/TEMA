"""Boundary cases shared by all future search and funding adapters."""

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
import app.pilot.multisource.contracts as signal_contracts

from app.pilot.multisource.contracts import (
    CapitalEvent, QueryProfile, SearchObservation, SignalFinding, SignalProfile,
    SourceSnapshot, TechnologyAssociation, load_policy,
)


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
H = "a" * 64
OTHER = "b" * 64


def search(**changes):
    values = dict(series_id=H, query_profile_hash=H, snapshot_hash=OTHER, phrase="молекулярная память",
                  phrase_role="technology", matching_mode="wordstat-monthly-v1", period_start=date(2026, 8, 1),
                  period_end=date(2026, 8, 31), is_complete_period=True, count=10, value_status="observed",
                  share_raw="0.5", share_unit="percent", share_fraction="0.005",
                  normalization_status="usable", observed_at=NOW, available_at=NOW, row_locator="row:2")
    values.update(changes)
    return SearchObservation(**values)


def grant(**changes):
    values = dict(event_id=uuid4(), source="cordis", source_hash=H, kind="grant_project", status="confirmed",
                  project_id="101234567", beneficiary_ids=("ORG1", "ORG2"), coordinator_id="ORG1",
                  agreement_at=date(2026, 8, 1), event_date_precision="day", observed_at=NOW,
                  available_at=NOW, amount="1000000", currency="EUR", amount_status="disclosed",
                  amount_kind="eu_contribution")
    values.update(changes)
    return CapitalEvent(**values)


def test_policy_is_pinned_and_extra_fields_fail(tmp_path):
    policy, digest = load_policy()
    assert policy.attention_limit == 15 and len(digest) == 64
    source = Path(signal_contracts.__file__).with_name("policy-v1.json")
    changed = tmp_path / "reformatted.json"
    changed.write_bytes(source.read_bytes() + b" ")
    assert load_policy(changed)[1] != digest
    path = tmp_path / "policy.json"
    path.write_text('{"version":"multisource/1.0.0","growth_threshold":"0.25","min_recent_count":30,'
                    '"min_persistence_numerator":2,"seasonal_lower":"-0.20","seasonal_upper":"0.25",'
                    '"spike_multiplier":"3","attention_limit":15,"concept_limit":100,'
                    '"wordstat_seed_limit":3,"wordstat_suggestion_limit":20,"wordstat_phrase_limit":15,'
                    '"wordstat_attempt_limit_per_run":40,"wordstat_attempt_limit_per_hour":80,"surprise":true}')
    with pytest.raises(ValidationError):
        load_policy(path)


def test_manual_scope_needs_no_scientific_plan_or_cloud_translation():
    profile = QueryProfile(profile_id=uuid4(), version=1, original_query="хранение данных в ДНК",
                           definition="Методы записи цифровых данных в молекулы ДНК")
    assert profile.scientific_plan_hash is None and profile.primary_phrase is None
    with pytest.raises(ValidationError):
        QueryProfile.model_validate(profile.model_dump() | {"version": True})
    with pytest.raises(ValidationError):
        QueryProfile.model_validate(profile.model_dump() | {"scientific_plan_hash": "not-a-hash"})


def test_observed_zero_is_distinct_from_missing_and_rounded_share():
    assert search(count=0, share_raw="0", share_fraction="0").count == 0
    assert search(count=None, value_status="missing", share_raw=None, share_fraction=None,
                  normalization_status="unknown_unit").count is None
    rounded = search(count=3, share_raw="0", share_fraction=None, normalization_status="quantized_zero")
    assert rounded.count == 3
    with pytest.raises(ValidationError):
        search(count=0, share_raw="0.5", share_fraction="0.005")
    with pytest.raises(ValidationError):
        search(count=3, share_raw="0", share_fraction="0", normalization_status="usable")
    with pytest.raises(ValidationError):
        search(count=0, value_status="missing")


def test_search_unit_dates_and_versions_do_not_silently_change():
    for changes in ({"share_fraction": "0.5"}, {"share_unit": "unknown"},
                    {"period_end": date(2026, 8, 30)}, {"count": True},
                    {"available_at": datetime(2026, 9, 20, tzinfo=timezone.utc)},
                    {"signal_schema_version": 2}, {"row_locator": "=SUM(1,2)\n"}):
        with pytest.raises(ValidationError):
            search(**changes)
    assert search(period_end=date(2026, 8, 30), is_complete_period=False).is_complete_period is False


def test_grant_is_one_event_with_multiple_beneficiaries():
    event = grant()
    assert event.project_id == "101234567" and event.beneficiary_ids == ("ORG1", "ORG2")
    with pytest.raises(ValidationError):
        grant(project_id=None)
    with pytest.raises(ValidationError):
        grant(beneficiary_ids=("ORG1", "ORG1"))
    with pytest.raises(ValidationError):
        grant(amount=None, amount_status="disclosed")
    with pytest.raises(ValidationError):
        grant(kind="equity_round", project_id=None, amount_kind="round_amount", recipient_name="Firm")


def test_unreported_amount_is_not_zero_or_fx_converted():
    event = grant(amount=None, currency="EUR", amount_status="undisclosed")
    assert event.amount is None and event.currency == "EUR"
    with pytest.raises(ValidationError):
        grant(amount="0", amount_status="undisclosed")
    with pytest.raises(ValidationError):
        grant(amount="NaN")
    with pytest.raises(ValidationError):
        grant(amount="1e9999999")


def test_confirmed_company_technology_link_requires_actual_evidence():
    values = dict(concept_id=uuid4(), subject_id="company-1", subject_kind="organisation",
                  relation="develops", status="confirmed", relation_at_event="supported",
                  reviewer="Analyst", reviewed_at=NOW, evidence_hashes=(H,))
    assert TechnologyAssociation(**values).attributable_amount is None
    with pytest.raises(ValidationError):
        TechnologyAssociation(**(values | {"evidence_hashes": ()}))
    with pytest.raises(ValidationError):
        TechnologyAssociation(**(values | {"attributable_amount": "500"}))


def test_profile_links_only_real_findings_and_rejects_wrong_queue():
    item = SignalFinding(finding_id=uuid4(), concept_id=uuid4(), origins=("wordstat",),
                         search_state="sustained_growth", observation_hashes=(H,), queue="attention",
                         rule_id="search_growth_v1", explanation="Наблюдаемый рост", next_check="Проверить термин")
    values = dict(profile_id=uuid4(), policy_hash=H, query_profile_hash=H, decision_at=NOW,
                  collection_finished_at=NOW, knowledge_cutoff=NOW, findings=(item,), attention_ids=(item.finding_id,))
    assert SignalProfile(**values).attention_ids == (item.finding_id,)
    with pytest.raises(ValidationError):
        SignalProfile(**(values | {"attention_ids": (), "watch_ids": (item.finding_id,)}))
    with pytest.raises(ValidationError):
        SignalProfile(**(values | {"attention_ids": ()}))
    with pytest.raises(ValidationError):
        SignalProfile(**(values | {"deferred_ids": (item.finding_id,)}))
    with pytest.raises(ValidationError):
        SignalProfile(**(values | {"knowledge_cutoff": datetime(2026, 9, 20, tzinfo=timezone.utc)}))


def test_source_rights_and_coverage_fail_closed():
    snapshot = dict(source="wordstat", adapter_version="wordstat-csv/1", request_hash=H,
                    query_profile_hash=H, observed_at=NOW, available_at=NOW,
                    coverage="partial", comparable=False, raw_hash=OTHER)
    assert SourceSnapshot(**snapshot).export_right == "unknown"
    with pytest.raises(ValidationError):
        SourceSnapshot(**(snapshot | {"comparable": True}))
    with pytest.raises(ValidationError):
        SourceSnapshot(**(snapshot | {"retention": "extract_only"}))
    with pytest.raises(ValidationError):
        SourceSnapshot(**(snapshot | {"source_url": "file:///tmp/secret"}))
    with pytest.raises(ValidationError):
        SourceSnapshot(**(snapshot | {"export_right": "share_allowed"}))
    assert SourceSnapshot(**(snapshot | {"export_right": "share_allowed",
                                     "license_ref": "dataset-license-2026"})).export_right == "share_allowed"

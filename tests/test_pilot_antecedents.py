"""Older coverage is bounded, reproducible, and does not imply novelty."""

from datetime import date

import pytest

from app.backend.contracts import SourcePage
from app.backend.errors import BackendError
from app.pilot.antecedents import AntecedentBundle, collect_antecedents, verify_antecedents
from app.pilot.evidence import EvidenceError
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled
from tests import test_pilot_history
from tests.test_pilot_evidence import Context, document, query_plan

scenario = test_pilot_history.scenario


class Provider:
    def __init__(self, documents=(), *, exhausted=True, skipped=0, error=None):
        self.documents = documents
        self.exhausted = exhausted
        self.skipped = skipped
        self.error = error
        self.requests = []
        self.closed = False

    def iter_pages(self, request, cancel):
        self.requests.append(request)
        if self.error:
            raise self.error
        yield SourcePage(documents=self.documents, scanned=len(self.documents) + self.skipped,
                         skipped=self.skipped, exhausted=self.exhausted)

    def close(self):
        self.closed = True


def bundle_for(scenario, documents=(), *, exhausted=True, context=None, **kwargs):
    archive, _, _, _, item, _, _ = scenario
    source = Provider(documents, exhausted=exhausted)
    bundle = collect_antecedents(item, query_plan(), archive, CredentialStore(), context or Context(),
        provider_factory=lambda _: source, **kwargs)
    return bundle, source


def test_older_collection_uses_public_dates_and_preserves_first_observation_source(scenario):
    old = (document(81, year=2015), document(82, year=2009))
    bundle, provider = bundle_for(scenario, old)
    assert provider.closed
    assert provider.requests[0].from_date == date(1900, 1, 1)
    assert provider.requests[0].until_date == date(2019, 12, 31)
    assert bundle.operational_status == "earlier_matches_found" and bundle.search_complete
    assert bundle.earliest_observed_year == 2009 and bundle.earliest_observed_study_id == old[1].document_key
    assert len(bundle.evidence) == 4
    assert any("не доказывает новизну" in value for value in bundle.limitations)
    archive, _, _, historical, item, _, _ = scenario
    verify_antecedents(bundle, item, query_plan(), archive)
    assert all(ref.publication_year >= 2020 for ref in historical.documents)


def test_complete_empty_query_does_not_become_novelty_and_incomplete_empty_is_explicit(scenario):
    complete, _ = bundle_for(scenario)
    assert complete.operational_status == "none_found_within_queries" and complete.search_complete
    incomplete, _ = bundle_for(scenario, exhausted=False)
    assert incomplete.operational_status == "incomplete_search" and not incomplete.search_complete


def test_title_rule_is_shared_and_old_versions_are_not_double_counted(scenario):
    old = document(81, year=2010)
    duplicate = type(old).model_validate(old.model_dump() | {"source_id": "W8888"})
    unrelated = document(82, year=2000, title="An unrelated water filter")
    bundle, _ = bundle_for(scenario, (old, duplicate, unrelated))
    assert bundle.matched_study_ids == (old.document_key,) and bundle.earliest_observed_year == 2010


def test_conflicting_dates_and_retracted_versions_cannot_hide_older_evidence(scenario):
    old = document(81, year=2010)
    conflict = type(old).model_validate(old.model_dump() | {"source_id": "W8888", "publication_year": 2011})
    bundle, _ = bundle_for(scenario, (old, conflict))
    assert not bundle.matched_study_ids and bundle.conflicting_study_ids == (old.document_key,)
    assert not bundle.search_complete and bundle.operational_status == "incomplete_search"
    retracted = type(old).model_validate(old.model_dump() | {"source_id": "W8888", "raw_metadata": {"is_retracted": True}})
    bundle, _ = bundle_for(scenario, (old, retracted))
    assert not bundle.matched_study_ids


def test_checkpoint_reuses_exact_archive_without_network(scenario):
    context = Context()
    bundle, _ = bundle_for(scenario, (document(81, year=2010),), context=context)
    archive, _, _, _, item, _, _ = scenario
    restored = collect_antecedents(item, query_plan(), archive, CredentialStore(), context,
        provider_factory=lambda _: pytest.fail("Checkpoint must not repeat network work"))
    assert restored == bundle and restored.bundle_hash == bundle.bundle_hash


def test_missing_api_and_record_budget_never_claim_complete_coverage(scenario):
    archive, _, _, _, item, _, _ = scenario
    source = Provider(error=BackendError("rate_limited", "Safe error"))
    bundle = collect_antecedents(item, query_plan(), archive, CredentialStore(), Context(),
        provider_factory=lambda _: source)
    assert bundle.operational_status == "incomplete_search"
    capped, _ = bundle_for(scenario, (document(81, year=2010),), max_documents=1, exhausted=False)
    assert capped.snapshot.coverage[0].limit_reached and not capped.search_complete


def test_tampered_first_observation_and_user_cancel_are_rejected(scenario):
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    forged = AntecedentBundle.model_validate(bundle.model_dump() | {"earliest_observed_year": 1901})
    archive, _, _, _, item, _, _ = scenario
    with pytest.raises(EvidenceError, match="не воспроизводится"):
        verify_antecedents(forged, item, query_plan(), archive)
    context = Context()
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        collect_antecedents(item, query_plan(), archive, CredentialStore(), context)

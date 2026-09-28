"""Real withdrawn preprint regression and conservative shared admission checks."""

from datetime import date
import json
from pathlib import Path

import pytest

from app.backend.contracts import DocumentRecord, SourcePage
from app.pilot.archive import DocumentArchive
from app.pilot.discovery import _eligible_studies, deduplicate, discover
from app.pilot.history import assess_snapshot, verify_history_artifact
from app.pilot.retractions import is_explicitly_retracted, retracted_family_keys
from app.pilot.signal_evidence import extract_signal_evidence
from app.pilot.sources import collect_snapshot, exclusion_reason
from app.runtime.credentials import CredentialStore
from tests.test_pilot_discovery import FixtureEncoder, plan as discovery_plan
from tests.test_pilot_evidence import document, query_plan, snapshot
from tests.test_pilot_history import scenario as scenario
from tests.test_pilot_signal_evidence import EXPERIMENT, NOVELTY, prepared
from tests.test_pilot_sources import Context, Provider, plan as source_plan


@pytest.mark.parametrize("changes", [
    {"title": "[Retracted] An experimental membrane mechanism"},
    {"title": "(WITHDRAWN) An experimental membrane mechanism"},
    {"title": "RETRACTED: An experimental membrane mechanism"},
    {"title": "Retracted Article: An experimental membrane mechanism"},
    {"title": "Retraction Note: An experimental membrane mechanism"},
    {"title": "An experimental membrane mechanism [Retracted]"},
    {"title": "An experimental membrane mechanism (withdrawn)"},
    {"title": "［Ｒｅｔｒａｃｔｅｄ］ An experimental membrane mechanism"},
    {"title": "<b>[Retracted]</b> An experimental membrane mechanism"},
    {"abstract": "This manuscript has been retracted."},
    {"abstract": "This article was retracted by the publisher in 2025."},
    {"abstract": "Abstract: This preprint is withdrawn due to errors."},
    {"abstract": None, "raw_metadata": {"abstract": "<jats:p>This manuscript has been retracted.</jats:p>"}},
    {"raw_metadata": {"is_retracted": True}},
    {"document_type": "retracted-article"},
    {"raw_metadata": {"type": "retraction"}},
])
def test_explicit_markers_have_the_same_admission_in_discovery_and_history(changes):
    record = document(1, **changes)
    assert is_explicitly_retracted(record)
    assert exclusion_reason(record, date(2020, 1, 1), date(2025, 12, 31)) == "retracted"
    result = discover([record], discovery_plan(), encoder=FixtureEncoder(), snapshot_id="withdrawn")
    assert result["unique_studies"] == 0 and not result["candidates"]
    assert result["excluded_studies"] == [{"study_id": record.document_key, "reason": "explicitly_retracted"}]


@pytest.mark.parametrize("changes", [
    {"title": "Retracted scientific papers: a bibliometric analysis"},
    {"title": "Retraction of scientific articles: causes and consequences"},
    {"title": "Retraction of biological tissue during mechanical measurement"},
    {"title": "Retraction notices and publication practices"},
    {"abstract": "We study why articles have been retracted and their citation impact."},
    {"abstract": 'We analyze notices saying "This manuscript has been retracted."'},
    {"abstract": '"This manuscript has been retracted." is a common publisher statement.'},
    {"abstract": "This article has not been retracted."},
    {"abstract": "This article may be retracted if its findings cannot be reproduced."},
    {"abstract": "This paper has been retracted? We investigate author notification."},
    {"raw_metadata": {"is_retracted": "false", "title": "[Retracted] A cited paper"}},
    {"raw_metadata": {"is_retracted": 1}},
    {"raw_metadata": {"references": [{"title": "[Retracted] A cited paper"}]}},
    {"raw_metadata": {"relation": {"has-review": [{"id-type": "doi", "id": "10.32388/btc1vy"}]}}},
])
def test_discussion_citation_negation_and_ambiguous_metadata_do_not_imply_own_withdrawal(changes):
    record = document(1, **changes)
    assert not is_explicitly_retracted(record)
    assert exclusion_reason(record, date(2020, 1, 1), date(2025, 12, 31)) is None


def test_actual_crossref_three_version_failure_is_closed_before_embedding():
    fixture = json.loads((Path(__file__).parent / "fixtures/pilot/retracted_nested_neural_networks.json").read_text(encoding="utf-8"))
    records = [DocumentRecord.model_validate(item) for item in fixture["documents"]]
    assert {record.doi for record in records} == {"10.32388/btc1vy", "10.32388/btc1vy.2", "10.32388/btc1vy.3"}
    families = deduplicate(records)
    assert len(families) == 1
    eligible, excluded = _eligible_studies(families, records, discovery_plan(), None)
    assert not eligible
    assert excluded == [{"study_id": families[0]["study_id"], "reason": "explicitly_retracted"}]
    # Removing either form of evidence still catches the other independently.
    for change in ({"title": "Nested Neural Networks: A Novel Approach to Flexible and Deep Learning Architectures"},
                   {"abstract": None, "raw_metadata": {}}):
        assert all(is_explicitly_retracted(record.model_copy(update=change)) for record in records)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("purpose", ["discovery", "history"])
@pytest.mark.parametrize("rules_version", ["publication-status/1.0.0", "publication-status/2.0.0"])
def test_collector_blocks_linked_clean_copy_even_when_withdrawal_is_discarded_first(tmp_path, reverse, purpose, rules_version):
    withdrawn = document(1, title="[Retracted] Original experimental manuscript", raw_metadata={"relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study2"}]}})
    clean = document(2)
    unrelated = document(3, raw_metadata={"relation": {"has-review": [{"id-type": "doi", "id": withdrawn.doi}]}})
    records = [withdrawn, clean, unrelated]
    if reverse:
        records.reverse()
    provider = Provider([SourcePage(documents=tuple(records), scanned=3, exhausted=True)])
    result = collect_snapshot(source_plan(purpose), Context(), DocumentArchive(tmp_path), CredentialStore(),
                              purpose=purpose, provider_factory=lambda _: provider, rules_version=rules_version)
    # v2 retains the rejected metadata for independent downstream verification.
    assert {reference.study_id for reference in result.documents} == (
        {item.document_key for item in records} if rules_version.endswith("/2.0.0") else {unrelated.document_key})
    assert result.coverage[0].accepted_records == 1 and result.coverage[0].rejected_records == 2
    assert result.coverage[0].scanned_records == 3 and result.coverage[0].state == "complete"


@pytest.mark.parametrize("sources", [("openalex", "crossref"), ("crossref", "openalex")])
@pytest.mark.parametrize("rules_version", ["publication-status/1.0.0", "publication-status/2.0.0"])
def test_text_only_withdrawal_blocks_same_doi_across_sources(tmp_path, sources, rules_version):
    def factory(source):
        item = document(1, source=source, abstract="This manuscript has been retracted." if source == "crossref"
                        else "A useful experimental manuscript.")
        return Provider([SourcePage(documents=(item,), scanned=1, exhausted=True)])

    result = collect_snapshot(source_plan(sources=sources), Context(), DocumentArchive(tmp_path), CredentialStore(),
                              provider_factory=factory, rules_version=rules_version)
    assert len(result.documents) == (2 if rules_version.endswith("/2.0.0") else 0)
    assert sum(item.accepted_records for item in result.coverage) == 0
    assert sum(item.rejected_records for item in result.coverage) == 2


def test_family_propagation_uses_version_links_not_shared_words_citations_or_authors():
    withdrawn = document(1, title="[Retracted] Selective membranes", raw_metadata={"relation": {
        "has-version": [{"id-type": "doi", "id": "10.1234/study2"}]}})
    second = document(2, raw_metadata={"relation": {"has-version": [{"id-type": "doi", "id": "10.1234/study3"}]}})
    third = document(3)
    unrelated = document(4, authors=withdrawn.authors, raw_metadata={"relation": {
        "has-review": [{"id-type": "doi", "id": withdrawn.doi}],
        "is-supplement-to": [{"id-type": "doi", "id": withdrawn.doi}]}})
    assert retracted_family_keys([withdrawn, second, third, unrelated]) == {
        withdrawn.document_key, second.document_key, third.document_key}


def test_text_only_withdrawal_cannot_manufacture_historical_growth(scenario):
    archive, docs, _, _, item, context, passport = scenario
    withdrawn = docs[-1].model_copy(update={"source_id": "W999999", "abstract": "This manuscript has been retracted."})
    historical = snapshot((*docs, withdrawn), archive, purpose="history")
    artifact, card, rejected = assess_snapshot(item, query_plan(), historical, archive, context,
                                              passport=passport, methodology_version="3.2.0")
    assert artifact.assessment.recent_studies == 13 and rejected == 1
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)
    # Frozen evaluators keep their original classification when replaying old files.
    for version in ("3.0.0", "3.1.0"):
        legacy, old_card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
                                             passport=passport, methodology_version=version)
        assert legacy.assessment.recent_studies == 14
        verify_history_artifact(legacy, query_plan(), historical, archive, old_card, context)


def test_withdrawn_article_cannot_supply_an_automatic_novelty_claim(tmp_path):
    archive, _, data, item, context = prepared(tmp_path, title="[Retracted] Lithium selective membranes",
        abstract=NOVELTY + " " + EXPERIMENT)
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan()) == ()


def test_later_retraction_is_not_injected_into_strict_historical_replay():
    from scripts.backtest_pilot import prepare_asof
    from tests.test_pilot_backtest import archived, configuration

    old = archived(1)
    withdrawn = archived(1, observed=2026, title="[Retracted] " + old.title,
                         abstract="This manuscript has been retracted.")
    before, _ = prepare_asof([old], configuration())
    after, report = prepare_asof([old, withdrawn], configuration())
    assert before == after and report["excluded"]["late_observation"] == 1
    assert not retracted_family_keys(after)


def test_antecedent_withdrawal_fix_preserves_original_bundle_replay(scenario, monkeypatch):
    from app.pilot.antecedents import (
        ANTECEDENT_VERSION, LEGACY_ANTECEDENT_VERSION, AntecedentBundle, verify_antecedents,
    )
    from tests.test_pilot_antecedents import bundle_for

    archive, _, _, _, item, _, _ = scenario
    withdrawn = document(81, year=2010, abstract="This manuscript has been retracted.")
    # Reconstruct the original v1.0 collector's explicit metadata-only rule.
    with monkeypatch.context() as old:
        old.setattr("app.pilot.antecedents.ANTECEDENT_VERSION", LEGACY_ANTECEDENT_VERSION)
        legacy, _ = bundle_for(scenario, (withdrawn,))
    assert legacy.version == LEGACY_ANTECEDENT_VERSION
    assert legacy.matched_study_ids == (withdrawn.document_key,)
    encoded, digest = legacy.model_dump_json(), legacy.bundle_hash
    verify_antecedents(legacy, item, query_plan(), archive)
    assert legacy.model_dump_json() == encoded and legacy.bundle_hash == digest
    # An old payload omitting its default version must stay on the old rule too.
    unspecified = AntecedentBundle.model_validate(legacy.model_dump(exclude={"version"}))
    assert unspecified == legacy
    verify_antecedents(unspecified, item, query_plan(), archive)
    current, _ = bundle_for(scenario, (withdrawn,))
    assert current.version == ANTECEDENT_VERSION and not current.matched_study_ids
    verify_antecedents(current, item, query_plan(), archive)
    assert current.bundle_hash != legacy.bundle_hash


def test_antecedent_rule_upgrade_does_not_reuse_old_checkpoint(scenario, monkeypatch):
    from app.pilot.antecedents import ANTECEDENT_VERSION, LEGACY_ANTECEDENT_VERSION
    from tests.test_pilot_antecedents import bundle_for
    from tests.test_pilot_evidence import Context as EvidenceContext

    context = EvidenceContext()
    withdrawn = document(81, year=2010, title="[Retracted] Lithium selective membranes")
    with monkeypatch.context() as old:
        old.setattr("app.pilot.antecedents.ANTECEDENT_VERSION", LEGACY_ANTECEDENT_VERSION)
        legacy, _ = bundle_for(scenario, (withdrawn,), context=context)
    current, provider = bundle_for(scenario, (withdrawn,), context=context)
    assert legacy.matched_study_ids and not current.matched_study_ids
    assert current.version == ANTECEDENT_VERSION and provider.requests
    assert len(context.checkpoints) == 2

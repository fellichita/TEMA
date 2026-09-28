"""Versioned necessary scope grounding: real MagLOV regression, no truth labels."""

import json
from pathlib import Path

import httpx
import pytest

from app.backend.contracts import DocumentRecord
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import (
    LEGACY_SCOPE_RULE_VERSION, SCOPE_RULE_VERSION, Candidate, QueryPlan, TrendCard, content_hash,
)
from app.pilot.evidence import (
    TITLE_ADMISSION_VERSION, admission_hash, build_passport, label_candidates, scope_is_anchored,
)
from app.pilot.signal_evidence import extract_signal_evidence, verify_signal_evidence, verify_signal_sources
from tests.test_pilot_evidence import Context, candidate, document, snapshot
from tests import test_pilot_llm

clients = test_pilot_llm.clients


def real_fixture(tmp_path):
    saved = json.loads((Path(__file__).parent / "fixtures/pilot_maglov_scope.json").read_text(encoding="utf-8"))
    plan = QueryPlan.model_validate(saved["query_plan"])
    doc = DocumentRecord.model_validate(saved["document"])
    archive = DocumentArchive(tmp_path / "revisions")
    data = snapshot((doc,), archive).model_copy(update={"plan_hash": plan.plan_hash})
    item = candidate(data, label=doc.title, scope_rule_version=SCOPE_RULE_VERSION)
    return plan, doc, archive, data, item


def test_actual_maglov_scope_has_local_query_concepts_but_no_legacy_exact_alias(tmp_path):
    plan, doc, archive, data, _ = real_fixture(tmp_path)
    members = ((data.documents[0], doc),)
    assert not scope_is_anchored(members, plan)
    assert scope_is_anchored(members, plan, rule_version=SCOPE_RULE_VERSION)
    # Grounding is insensitive to source identity; no known DOI override exists.
    anonymous = doc.model_copy(update={"doi": None, "source_id": "W999"})
    assert scope_is_anchored(((archive.put(anonymous), anonymous),), plan, rule_version=SCOPE_RULE_VERSION)


@pytest.mark.parametrize("abstract", [
    "Quantum computing algorithms for biological systems.",
    "Classical fluorescence sensing of biological systems.",
    "Quantum sensing of physical systems.",
    "Quantum sensing in biological samples.",
    "Quantum " + "context " * 80 + "sensing in biological systems.",
    "Quantum phenomena.\n\nSensing in biological systems.",
    "Quantum phenomena.\nSensing in biological systems.",
])
def test_missing_technical_concept_or_nonlocal_cooccurrence_does_not_ground_scope(tmp_path, abstract):
    plan, _, archive, _, _ = real_fixture(tmp_path)
    # Remove aliases to test the new fallback alone, including sensing/system.
    plan = plan.model_copy(update={"synonyms": (), "subdirections": ()})
    doc = document(1, title="Experimental measurement", abstract=abstract)
    assert not scope_is_anchored(((archive.put(doc), doc),), plan, rule_version=SCOPE_RULE_VERSION)


def test_local_scope_preserves_plural_concepts_exact_alias_and_document_boundaries(tmp_path):
    plan, _, archive, _, _ = real_fixture(tmp_path)
    plural = document(1, title="Experimental measurement", abstract="Sensing a biological system through quantum phenomena.")
    alias = document(2, title="Quantum biosensing by spin resonance", abstract=None)
    assert scope_is_anchored(((archive.put(plural), plural), (archive.put(alias), alias)), plan,
                             rule_version=SCOPE_RULE_VERSION)
    first = document(3, title="Quantum phenomena", abstract="Experimental measurement.")
    second = document(4, title="Sensing of biological systems", abstract=None)
    assert not scope_is_anchored(((archive.put(first), first), (archive.put(second), second)), plan,
                                 rule_version=SCOPE_RULE_VERSION)
    mixed = document(5, title="Classical fluorescence sensing of biological systems", abstract=None)
    assert not scope_is_anchored(((archive.put(alias), alias), (archive.put(mixed), mixed)), plan,
                                 rule_version=SCOPE_RULE_VERSION)
    assert not scope_is_anchored((), plan, rule_version=SCOPE_RULE_VERSION)


def test_saved_32_signal_evidence_replays_legacy_rule_and_new_rule_is_independently_reproducible(tmp_path):
    plan, _, archive, _, _ = real_fixture(tmp_path)
    doc = document(3, title="Spin resonance protein sensors",
        abstract="Sensing a biological system uses quantum spin phenomena. "
            "Here we introduce novel spin resonance protein sensors with a tethered spin transport mechanism. "
            "We measured magnetic response in laboratory experiments.")
    data = snapshot((doc,), archive).model_copy(update={"plan_hash": plan.plan_hash})
    legacy = candidate(data, label=doc.title, specificity="specific_technology",
        synonyms=("spin resonance protein sensors",), admission_rule_version=TITLE_ADMISSION_VERSION)
    legacy = legacy.model_copy(update={"admission_rule_hash": admission_hash(legacy)})
    assert extract_signal_evidence(legacy, data, archive, Context(), query_plan=plan) == ()
    verify_signal_evidence((), legacy, data, archive, Context(), query_plan=plan)
    current = legacy.model_copy(update={"scope_rule_version": SCOPE_RULE_VERSION})
    current = current.model_copy(update={"admission_rule_hash": admission_hash(current)})
    evidence = extract_signal_evidence(current, data, archive, Context(), query_plan=plan)
    assert len(evidence) == 1
    verify_signal_evidence(evidence, current, data, archive, Context(), query_plan=plan)
    verify_signal_sources(evidence, current, archive, Context(), query_plan=plan)
    # Updating the program must not retroactively add evidence to old 3.2 cards.
    assert extract_signal_evidence(Candidate.model_validate(legacy.model_dump()), data, archive,
                                   Context(), query_plan=plan) == ()


@pytest.mark.parametrize("proposal_kind", ["supported", "off_scope", "missing_quote"])
def test_actual_maglov_naming_still_requires_llm_scope_evidence_and_does_not_prove_weak_signal(tmp_path, clients, proposal_kind):
    plan, doc, archive, data, item = real_fixture(tmp_path)
    proposal = {"candidate_id": item.candidate_id, "label": "Спиновый резонанс в инженерных белках",
        "definition": "Quantum spin resonance in engineered proteins for sensing.",
        "phrases": ["Quantum spin resonance in engineered proteins"], "exclusions": [],
        "specificity": "specific_technology", "in_scope": proposal_kind != "off_scope",
        "scope_support": [] if proposal_kind == "missing_quote" else [{
            "revision_id": data.documents[0].revision_id, "field": "abstract",
            "quote": "We find that MagLOV exhibits optically detected magnetic resonance in living bacterial cells at room temperature, at sufficiently high signal-to-noise for single-cell detection."}],
        "scope_reason": "Experimental sensing mechanism in living cells."}
    client, budget = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps({"candidates": [proposal]}))))
    context = Context()
    frozen = label_candidates((item,), data, archive, context, client=client,
                              scope_ids=("run:one",), query_plan=plan)
    assert budget.snapshot("run:one").used.calls == 1
    if proposal_kind != "supported":
        assert frozen == ()
        return
    checked, = frozen
    assert checked.specificity == "specific_technology"
    assert checked.scope_rule_version == SCOPE_RULE_VERSION
    assert checked.admission_rule_hash == admission_hash(checked)
    card = build_passport(checked, data, archive, Context())
    assert card.category == "insufficient_evidence" and card.assessment_hash is None
    assert extract_signal_evidence(checked, data, archive, Context(), query_plan=plan) == ()
    verify_signal_evidence((), checked, data, archive, Context(), query_plan=plan)
    assert label_candidates((checked,), data, archive, Context(), query_plan=plan) == (checked,)


@pytest.mark.parametrize("methodology", ["3.0.0", "3.1.0", "3.2.0"])
@pytest.mark.parametrize("admission", ["title-phrase-admission/1.0.0", TITLE_ADMISSION_VERSION])
def test_legacy_candidate_and_card_hashes_are_unchanged_and_new_rule_is_explicit(tmp_path, methodology, admission):
    _, _, _, data, item = real_fixture(tmp_path)
    legacy = item.model_copy(update={"scope_rule_version": LEGACY_SCOPE_RULE_VERSION,
                                    "admission_rule_version": admission})
    payload = legacy.model_dump(mode="json")
    assert "scope_rule_version" not in payload
    roundtrip = Candidate.model_validate(payload)
    assert roundtrip.scope_rule_version == LEGACY_SCOPE_RULE_VERSION
    assert content_hash(roundtrip) == content_hash(payload)
    assert admission_hash(legacy) == admission_hash(roundtrip)
    old_card = {"schema_version": 3, "methodology_version": methodology, "candidate": payload,
        "category": "unassessed_cluster", "quality": "partial", "claims": [], "evidence": [],
        "historical_snapshot_id": None, "assessment_hash": None, "limitations": ["History has not been assessed."]}
    # Exact omission of the new default survives nested result serialization.
    card = TrendCard.model_validate(old_card)
    serialized = card.model_dump(mode="json")
    assert serialized["candidate"] == payload
    assert content_hash(card) == content_hash(serialized)
    changed = roundtrip.model_copy(update={"scope_rule_version": SCOPE_RULE_VERSION})
    assert changed.model_dump(mode="json")["scope_rule_version"] == SCOPE_RULE_VERSION
    assert admission_hash(changed) != admission_hash(roundtrip)
    assert changed.discovery_snapshot_id == data.snapshot_id

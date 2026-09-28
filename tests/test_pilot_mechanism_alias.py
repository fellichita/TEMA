"""New naming must produce reusable mechanism queries, not copied paper titles."""

import json

import httpx
import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.evidence import (
    LABEL_PROMPT_VERSION, CandidateName, MechanismAliasRequired, _freeze, admission_hash,
    build_passport, label_candidates,
)
from tests import test_pilot_llm
from tests.test_pilot_evidence import Context, candidate, document, snapshot
from tests.test_pilot_scope_grounding import real_fixture

clients = test_pilot_llm.clients

WIGNER_TITLE = "Physiological Search for Quantum Biological Sensing Effects Based on the Wigner–Yanase Connection between Coherence and Uncertainty"
WIGNER_ABSTRACT = (
    "Abstract A fundamental concept of quantum physics, the Wigner–Yanase information, is used here as a measure "
    "of quantum coherence in spin‐dependent radical‐pair reactions pertaining to biological magnetic sensing. "
    "This measure is connected to the uncertainty of the reaction yields and, further, to the statistics of a "
    "cellular receptor‐ligand system used to biochemically convey magnetic‐field changes. Measurable physiological "
    "quantities, such as the number of receptors and fluctuations in ligand concentration, are shown to reflect "
    "the introduced Wigner–Yanase measure of singlet‐triplet coherence. A quantum‐biological uncertainty relation "
    "connecting the product of a biological resource and a biological figure of merit with the Wigner–Yanase "
    "coherence is arrived at. This approach can serve as a general search for quantum‐coherent effects within "
    "cellular environments."
)


def name(item, phrase, *, label="Конкретный механизм", support=()):
    return CandidateName(candidate_id=item.candidate_id, label=label,
        definition="A mechanism proposal, not an assertion of experimental replication or emergence.",
        phrases=(phrase,), specificity="specific_technology", in_scope=True, scope_support=support)


@pytest.mark.parametrize("phrase_kind", ["complete_paper_title", "canonical_mechanism"])
def test_actual_maglov_requires_canonical_history_phrase_without_losing_the_mechanism(tmp_path, clients, phrase_kind):
    plan, doc, archive, data, item = real_fixture(tmp_path)
    phrase = doc.title if phrase_kind == "complete_paper_title" else "Quantum spin resonance in engineered proteins"
    proposal = name(item, phrase, support=({"revision_id": data.documents[0].revision_id, "field": "title",
                                           "quote": doc.title},))
    calls = []

    def handler(request):
        calls.append(request)
        prompt = json.loads(request.content)["messages"][0]["content"]
        assert "concise canonical technical phrase" in prompt
        assert "theoretical model" in prompt and "never describe it as a demonstrated sensor" in prompt
        return httpx.Response(200, json=test_pilot_llm.response_payload(
            json.dumps({"candidates": [proposal.model_dump(mode="json")]})))

    client, budget = clients(handler)
    context = Context()
    checked, = label_candidates((item,), data, archive, context, client=client,
                                scope_ids=("run:one",), query_plan=plan)
    checkpoint = context.checkpoints["labels_0"]
    assert LABEL_PROMPT_VERSION == "candidate-labels/3.3.0"
    assert len(calls) == budget.snapshot("run:one").used.calls == 1
    if phrase_kind == "complete_paper_title":
        assert checked.specificity == "uncertain"
        assert checkpoint["failure_code"] == "mechanism_alias_required"
        assert "короткое название конкретного механизма" in checkpoint["limitation"]
        assert checkpoint["fallback_candidate_ids"] == [item.candidate_id]
    else:
        assert checked.specificity == "specific_technology"
        assert checked.synonyms == (phrase,)
        assert not checkpoint["fallback_candidate_ids"] and checkpoint["failure_code"] is None
    assert checked.admission_rule_hash == admission_hash(checked)
    assert build_passport(checked, data, archive, Context()).category in {"unassessed_cluster", "insufficient_evidence"}
    assert label_candidates((item,), data, archive, context, client=client,
                            scope_ids=("run:one",), query_plan=plan) == (checked,)
    assert len(calls) == 1  # No automatic paid repair of an unusable proposal.


def test_actual_wigner_yanase_paper_title_is_not_a_stable_mechanism_alias(tmp_path, clients):
    plan, base, archive, _, _ = real_fixture(tmp_path)
    doc = base.model_copy(update={"doi": "10.1002/qute.202300292", "source_id": "W-wigner-fixture",
        "title": WIGNER_TITLE, "abstract": WIGNER_ABSTRACT})
    data = snapshot((doc,), archive).model_copy(update={"plan_hash": plan.plan_hash})
    item = candidate(data, label=doc.title, scope_rule_version="all-query-concepts-local/2")
    proposal = name(item, doc.title, label="Теоретическая модель квантово-биологического зондирования",
        support=({"revision_id": data.documents[0].revision_id, "field": "title", "quote": doc.title},))
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(
        json.dumps({"candidates": [proposal.model_dump(mode="json")]}))))
    context = Context()
    checked, = label_candidates((item,), data, archive, context, client=client,
                                scope_ids=("run:one",), query_plan=plan)
    assert checked.specificity == "uncertain"
    assert context.checkpoints["labels_0"]["failure_code"] == "mechanism_alias_required"
    card = build_passport(checked, data, archive, Context())
    assert card.category == "unassessed_cluster"
    assert not any(claim.role == "application" and claim.support == "supported" for claim in card.claims)


@pytest.mark.parametrize("title", [
    "Photonic memory",
    "Photonic memory for optical computing",
    "Phase coherent genetically encoded diamond nitrogen vacancy spin ensemble magnetometry",
])
def test_short_legitimate_titles_and_long_mechanism_names_remain_eligible(tmp_path, title):
    archive = DocumentArchive(tmp_path / "revisions")
    doc = document(1, title=title)
    data = snapshot((doc,), archive)
    item = candidate(data)
    checked = _freeze(item, name(item, title), ((data.documents[0], doc),), current_proposal=True)
    assert checked.specificity == "specific_technology" and checked.synonyms == (title,)


def test_gate_does_not_reinterpret_frozen_historical_title_rules(tmp_path):
    plan, doc, archive, data, item = real_fixture(tmp_path)
    proposal = name(item, doc.title)
    members = ((data.documents[0], doc),)
    # Explicitly emulate a saved rule produced before the new prompt admission gate.
    old = _freeze(item, proposal, members)
    assert old.specificity == "specific_technology"
    old_hash = admission_hash(old)
    with pytest.raises(MechanismAliasRequired):
        _freeze(item, proposal, members, current_proposal=True)
    context = Context()
    preserved, = label_candidates((old,), data, archive, context, query_plan=plan)
    assert preserved == old and admission_hash(preserved) == old_hash
    assert context.checkpoints["labels_0"]["label_status"] == "retained_frozen_definition"
    assert context.checkpoints["labels_0"]["fallback_candidate_ids"] == []

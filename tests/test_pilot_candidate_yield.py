"""Why the TOP stayed empty: candidates the local model could not name, and none to name.

Measured on «solid-state batteries» (24.09.2026): 22 of 40 local namings failed the
answer schema, every density cluster was either tiny or the whole direction, and
the attempt pool was filled with single papers. These checks pin the repairs.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.pilot.archive import DocumentArchive
from app.pilot.evidence import LocalCandidateName, LocalLabelBatch, covering_title_phrases, label_candidates
from app.pilot.hierarchy import title_phrase_groups
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot

REVISION = "a" * 64


def answer(**changes):
    return {"candidate_id": "c0", "label": "Сульфидные электролиты", "definition": "Твёрдый сульфидный электролит.",
            "phrase_number": 1, "specificity": "specific_technology", "in_scope": True,
            "scope_support": [{"revision_id": REVISION, "field": "abstract", "quote": "One sentence."}],
            "scope_reason": "", **changes}


def test_small_model_formatting_slips_are_repaired_without_accepting_anything_unverified():
    long_quote = "Sulfide electrolytes conduct lithium ions. " + "More context follows here. " * 30
    repaired = LocalCandidateName.model_validate(answer(
        definition=None,
        scope_support=[{"revision_id": REVISION, "field": "abstract", "quote": long_quote,
                        "scope_reason": "The abstract is about sulfide electrolytes."},
                       {"revision_id": "the 64-character id supplied with that document", "field": "title",
                        "quote": "Polymer membrane gas separation at high pressure"}]))
    assert [item.quote for item in repaired.scope_support] == ["Sulfide electrolytes conduct lithium ions."]
    assert repaired.definition == "Sulfide electrolytes conduct lithium ions."
    assert repaired.scope_reason == "The abstract is about sulfide electrolytes."
    spaced = answer()
    spaced[" scope_reason"] = spaced.pop("scope_reason")
    assert LocalCandidateName.model_validate(spaced).scope_reason == ""


def test_a_copy_of_the_formatting_example_is_retried_not_read_as_a_rejection():
    with pytest.raises(ValidationError, match="formatting example"):
        LocalLabelBatch.model_validate({"candidates": [answer(
            label="polymer membrane gas separation", in_scope=False,
            definition="Полимерная мембрана разделяет газовую смесь без охлаждения.")]})


def test_title_phrase_groups_split_a_whole_direction_into_admissible_technologies():
    titles = ([f"Sulfide solid electrolytes for all-solid-state batteries {number}" for number in range(4)]
              + [f"Halide solid electrolytes with high voltage stability {number}" for number in range(3)]
              + ["Recent advances in solid-state batteries"] * 6
              + ["Solid-state batteries: challenges and prospects"] * 6
              + ["Can they really change the game for vehicles"] * 5)
    eligible = [True] * len(titles)
    eligible[0] = False
    groups = title_phrase_groups(titles, eligible, scope_names=("solid-state batteries",),
                                 minimum_size=3, maximum_share=0.5)
    by_phrase = {group["phrase"]: group["members"] for group in groups}
    # The group follows the shared phrase and admits only in-scope studies.
    assert by_phrase["sulfide solid electrolytes"] == [1, 2, 3]
    assert by_phrase["halide solid electrolytes"] == [4, 5, 6]
    # The direction's own name, rhetoric and clause fragments never form a group.
    assert all("solid state batteries" != phrase and "advances" not in phrase and "challenges" not in phrase
               and "they" not in phrase for phrase in by_phrase)


def test_offered_names_are_short_named_mechanisms_not_title_fragments():
    docs = tuple((None, document(number, title=title)) for number, title in enumerate((
        "Solid-state Li metal batteries with polymer electrolyte",
        "Dendrite-free solid-state Li metal batteries",
        "Protective interlayers for solid-state Li metal batteries")))
    offered = covering_title_phrases(docs, excluded=("solid-state batteries",), limit=6)
    assert offered[0] in {"li metal", "li metal batteries"}
    assert all(len(phrase.split()) <= 4 for phrase in offered)
    assert "solid state" not in offered and "solid state li" not in offered[:1]
    single = ((None, document(9, title="Solid-State Batteries — Can They Really Change the Game?")),)
    assert not any({"can", "they", "really"} & set(phrase.split())
                   for phrase in covering_title_phrases(single, excluded=("solid-state batteries",)))


class LocalClient:
    config = SimpleNamespace(provider="local")

    def __init__(self, respond=None):
        self.calls = 0
        self.respond = respond
        self.last_receipt = None

    def generate_json(self, schema, **_kwargs):
        self.calls += 1
        if self.respond is None:
            raise AssertionError("the generation was expected to be skipped")
        return SimpleNamespace(value=schema.model_validate(self.respond()))


def test_a_group_without_a_shared_title_phrase_is_not_sent_to_the_model(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1, title="Graphene oxide selective membranes for lithium brines"),
            document(2, title="Ion imprinted polymer selective membranes"))
    data = snapshot(docs, archive)
    context, client = Context(), LocalClient()
    label_candidates((candidate(data),), data, archive, context, client=client, query_plan=query_plan())
    assert client.calls == 0
    assert context.checkpoints["labels_0"]["failure_code"] == "evidence_rejected"


def test_a_title_quoted_as_abstract_is_found_in_the_title(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1),)
    data = snapshot(docs, archive)
    reference = data.documents[0]
    client = LocalClient(lambda: {"candidates": [answer(
        label="Литий-селективные мембраны", definition="Мембрана извлекает литий.",
        scope_support=[{"revision_id": reference.revision_id, "field": "abstract", "quote": docs[0].title}])]})
    context = Context()
    named = label_candidates((candidate(data),), data, archive, context, client=client, query_plan=query_plan())
    assert client.calls == 1
    assert context.checkpoints["labels_0"]["rejected"] == []
    assert context.checkpoints["labels_0"]["label_status"] == "model_proposal"
    assert named[0].specificity == "specific_technology"

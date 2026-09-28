"""TOP membership, stable ordering and retained refinement provenance."""

from random import Random

import pytest

from app.pilot.contracts import content_hash
from app.pilot.selection import select_top
from app.pilot.service import _history_subbudgets, _retained_snapshots
from tests.test_pilot_result_signals import automatic_result


def ranked_copy(card, artifact, index, *, name=None, aliases=None, priority=None):
    """Selection boundary only: scientific arithmetic is independently replay-tested."""
    candidate = card.candidate.model_copy(update={"candidate_id": f"mechanism-{index:03d}",
        "label": name or f"Mechanism design {index}", "synonyms": aliases or (f"mechanism design {index}",)})
    assessment = artifact.assessment.model_copy(update={"candidate_id": candidate.candidate_id,
        "signal_priority": float(index) if priority is None else priority})
    return (card.model_copy(update={"candidate": candidate, "assessment_hash": assessment.assessment_hash}),
            artifact.model_copy(update={"assessment": assessment}))


def test_top_fifteen_is_stable_under_input_order_and_preserves_unselected_cards(tmp_path):
    result, _, original = automatic_result(tmp_path)
    pairs = [ranked_copy(result.cards[0], original, index) for index in range(25)]
    cards, artifacts = map(list, zip(*pairs, strict=True))
    expected = tuple(f"mechanism-{index:03d}" for index in range(24, 9, -1))
    assert select_top(cards, artifacts) == expected
    Random(91).shuffle(cards)
    Random(92).shuffle(artifacts)
    assert select_top(cards, artifacts) == expected
    assert len(cards) == len(artifacts) == 25
    assert select_top(cards, artifacts, limit=5) == expected[:5]


def test_full_name_alias_duplicates_merge_but_distinct_sibling_mechanisms_remain(tmp_path):
    result, _, original = automatic_result(tmp_path)
    pairs = [ranked_copy(result.cards[0], original, 1, name="Lithium selective membranes",
                         aliases=("lithium selective membranes",)),
             ranked_copy(result.cards[0], original, 2, name="Другая формулировка",
                         aliases=("lithium selective membrane",)),
             ranked_copy(result.cards[0], original, 3, name="Sodium selective membranes",
                         aliases=("sodium selective membranes",))]
    cards, artifacts = zip(*pairs, strict=True)
    assert select_top(cards, artifacts) == ("mechanism-003", "mechanism-002")
    # A full label matching another candidate's alias is also one concept.
    altered = cards[0].model_copy(update={"candidate": cards[0].candidate.model_copy(update={
        "label": "Другая формулировка", "synonyms": ("independent naming here",)})})
    assert select_top((altered, *cards[1:]), artifacts) == ("mechanism-003", "mechanism-002")


def test_unknown_missing_mismatched_and_established_candidates_cannot_pad_top(tmp_path):
    result, _, artifact = automatic_result(tmp_path)
    card = result.cards[0]
    assert select_top((card,), ()) == ()
    assert select_top((card.model_copy(update={"assessment_hash": "f" * 64}),), (artifact,)) == ()
    for category in ("unassessed_cluster", "insufficient_evidence", "declining", "transient_burst", "established_topic"):
        assert select_top((card.model_copy(update={"category": category}),), (artifact,)) == ()
    assessment = artifact.assessment.model_copy(update={"signal_priority": None})
    assert select_top((card.model_copy(update={"assessment_hash": assessment.assessment_hash}),),
                      (artifact.model_copy(update={"assessment": assessment}),)) == ()


@pytest.mark.parametrize("limit", (0, 16, True, 2.5))
def test_top_rejects_invalid_limits(limit):
    with pytest.raises(ValueError):
        select_top((), (), limit=limit)


def test_repeated_refinement_prunes_superseded_snapshots_and_keeps_live_provenance(tmp_path):
    result, _, artifact = automatic_result(tmp_path)
    historical = result.snapshots[1]
    superseded = tuple(historical.model_copy(update={"snapshot_id": content_hash({"superseded": i})})
                       for i in range(120))
    retained = _retained_snapshots((*superseded, *result.snapshots), result.cards, (), (artifact,))
    assert {item.snapshot_id for item in retained} == {item.snapshot_id for item in result.snapshots}
    assert len(retained) == 3
    # Removing an optional redundant snapshot must not drop evidence-only provenance.
    without_inputs = _retained_snapshots(result.snapshots, result.cards, (), ())
    assert result.snapshots[0] in without_inputs and historical in without_inputs


def test_empty_result_keeps_discovery_snapshot_for_inspection(tmp_path):
    result, _, _ = automatic_result(tmp_path)
    assert _retained_snapshots(result.snapshots, (), (), ()) == (result.snapshots[0],)


def test_automatic_prior_search_and_history_cannot_exceed_allocated_document_budget():
    for allocation in (0, 1, 2, 3, 10, 500, 666, 3000):
        earlier, recent = _history_subbudgets(allocation)
        assert 0 <= earlier <= 500 and recent >= 0
        assert earlier + recent <= allocation
        if allocation >= 2:
            assert earlier > 0 and recent > 0 and earlier + recent == allocation

"""Mechanism review must not depend on a density algorithm's noise assignment."""

import numpy as np
import pytest

from app.pilot.discovery import DiscoveryOptions, discover
from app.pilot.hierarchy import refine_groups, study_hypotheses, study_review_priority
from app.pilot.service import _candidate_pool
from tests.test_pilot_discovery import FixtureEncoder, document, plan


class SameEncoder(FixtureEncoder):
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


@pytest.mark.parametrize("assigned", [True, False])
def test_individual_mechanism_survives_unsplittable_parent_and_noise(monkeypatch, assigned):
    records = [document(index, title=f"Review of sensor platforms {index}") for index in range(39)]
    target = document(39, title="Spin resonance in engineered proteins for multimodal sensing")
    records.append(target)
    monkeypatch.setattr("app.pilot.discovery._labels", lambda *args: (
        np.full(len(records), 0 if assigned else -1), "hdbscan", {"nmf": {"converged": True}}))
    result = discover(records, plan(), encoder=SameEncoder(), snapshot_id="s")
    if assigned:
        assert result["hierarchy"][0]["split"] is False
    standalone = [c for c in result["review_queue"] if c["discovery_study_ids"] == [target.document_key]]
    assert len(standalone) == 1
    assert standalone[0]["specificity"] == "uncertain"
    assert "category" not in standalone[0]
    metadata = next(m for m in result["review_queue_metadata"] if m["candidate_id"] == standalone[0]["candidate_id"])
    assert metadata["origin"] == ("cluster_member_study" if assigned else "unclustered_study")
    assert metadata["parent_node_id"] == (0 if assigned else None)
    assert standalone[0]["candidate_id"] in {c.candidate_id for c in _candidate_pool(result, 2)}
    assert len(result["review_queue"]) == len(records)


def test_child_hypothesis_identity_is_stable_when_parent_assignment_changes(monkeypatch):
    records = [document(index) for index in range(40)]
    results = []
    for label in [0, -1]:
        monkeypatch.setattr("app.pilot.discovery._labels", lambda *args, label=label: (
            np.full(len(records), label), "hdbscan", {"nmf": {"converged": True}}))
        results.append(discover(records, plan(), encoder=SameEncoder(), snapshot_id="s"))
    assert {c["candidate_id"] for c in results[0]["review_queue"]} == {
        c["candidate_id"] for c in results[1]["review_queue"]}


def test_every_assigned_study_keeps_a_review_path_when_group_cap_is_reached(monkeypatch):
    records = [document(index) for index in range(40)]
    monkeypatch.setattr("app.pilot.discovery._labels", lambda *args: (
        np.repeat(np.arange(10), 4), "hdbscan", {"nmf": {"converged": True}}))
    result = discover(records, plan(), encoder=SameEncoder(), snapshot_id="s",
                      options=DiscoveryOptions(maximum_candidates=1))
    singles = [c for c in result["review_queue"] if len(c["discovery_study_ids"]) == 1]
    assert len(singles) == 40
    assert len(result["review_queue"]) == 49  # Nine overflow groups are preserved too.
    pool = _candidate_pool(result, 6)
    assert [len(c.discovery_study_ids) for c in pool] == [4, 1, 4, 1, 4, 1]
    assert len({c.candidate_id for c in pool}) == 6


def test_a_one_study_leaf_does_not_duplicate_its_candidate():
    leaves = [{"members": [0], "node_id": 0, "depth": 0},
              {"members": [1, 2], "node_id": 1, "depth": 1}]
    hypotheses = study_hypotheses(leaves, 4)
    assert [group["members"] for group in hypotheses] == [[1], [2], [3]]
    assert [group["parent_node_id"] for group in hypotheses] == [1, 1, None]


@pytest.mark.parametrize("title,kind,expected", [
    ("Review—Quantum Biosensors: Principles and Applications", "article", 1),
    ("A contemporary overview of sensors", "article", 1),
    ("Обзор технологий измерения", "article", 1),
    ("Sensors", "book-chapter", 1),
    ("Sensors", "review", 1),
    ("Engineered protein spin resonance", "article", 0),
    ("Engineered protein spin resonance", "preprint", 0),
    ("Previewing the sensor response", "article", 0),
])
def test_synthesis_priority_is_only_an_explicit_publication_role(title, kind, expected):
    assert study_review_priority(document(1, title=title, raw_metadata={"type": kind})) == expected


def test_input_order_and_source_copies_cannot_multiply_paper_hypotheses(monkeypatch):
    records = [document(index, doi=f"10.1234/work{index}") for index in range(40)]
    records.append(document(100, doi="10.1234/work1", source="crossref"))
    monkeypatch.setattr("app.pilot.discovery._labels", lambda vectors, *args: (
        np.zeros(len(vectors), dtype=int), "hdbscan", {"nmf": {"converged": True}}))
    first = discover(records, plan(), encoder=SameEncoder(), snapshot_id="s")
    second = discover(list(reversed(records)), plan(), encoder=SameEncoder(), snapshot_id="s")
    assert len(first["review_queue"]) == 40
    assert [c["candidate_id"] for c in first["review_queue"]] == [
        c["candidate_id"] for c in second["review_queue"]]


def test_normalized_review_type_has_priority_even_without_raw_metadata():
    assert study_review_priority(document(1, document_type="review")) == 1


def test_strongly_unbalanced_maximum_corpus_refinement_preserves_rare_child():
    vectors = np.zeros((5000, 384), dtype=np.float32)
    vectors[:4998, 0], vectors[4998:, 1] = 1, 1
    leaves, nodes = refine_groups([list(range(5000))], vectors)
    assert nodes[0]["split"] is True
    assert {tuple(leaf["members"]) for leaf in leaves} == {tuple(range(4998)), (4998, 4999)}


def test_refinement_checks_cancellation_after_clustering(monkeypatch):
    from concurrent.futures import CancelledError
    from threading import Event

    event = Event()

    def cancel_after_fit(self, vectors):
        event.set()
        return np.repeat([0, 1], len(vectors) // 2)

    monkeypatch.setattr("sklearn.cluster.KMeans.fit_predict", cancel_after_fit)
    vectors = np.zeros((48, 384), dtype=np.float32)
    vectors[:24, 0], vectors[24:, 1] = 1, 1
    with pytest.raises(CancelledError):
        refine_groups([list(range(48))], vectors, cancel=event)

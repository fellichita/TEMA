"""End-to-end properties of the local candidate pipeline."""

from concurrent.futures import CancelledError
from copy import deepcopy
from datetime import date
import json
from threading import Event

import pytest

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.backend.config import BackendSettings
from app.backend.history import HistoryRequest
from app.backend.service import Backend
from app.ml.corpus import read_history, read_snapshot, unpack_snapshot
from app.ml.engine import analyze, metrics, prepare
from app.ml.service import export_result
from app.ml.text import evidence_card
from tests.mvp_fixture import Provider, snapshot, research_groups


OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


def test_pipeline_is_offline_reproducible_and_every_card_passage_is_from_its_source(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("No network in local analysis"))
    corpus = unpack_snapshot(snapshot())
    first = analyze(corpus, OPTIONS)
    second = analyze(corpus, OPTIONS)
    assert first == second
    assert 2 <= len(research_groups(first)) <= 15
    studies, _ = prepare(corpus["entries"], OPTIONS)
    by_id = {s["id"]:s for s in studies}
    assigned = []
    for candidate in research_groups(first):
        assert all(candidate["card"].values())
        assert 0 <= candidate["metrics"]["score"] <= 100
        assert candidate["stage"] == "requires_review"
        assigned.extend(candidate["study_ids"])
        for row in candidate["metrics"]["years"]:
            assert row["documents"] <= row["direction_documents"]
        for field in candidate["card"].values():
            if field:
                study = by_id[field["study_id"]]
                assert field["url"] == study["url"]
                assert field["text"] in (study["abstract"] if field["mode"] == "source_excerpt" else study["title"])
    assert len(assigned) == len(set(assigned))
    json.dumps(first, allow_nan=False)


def test_conflicting_years_do_not_silently_move_to_latest_date():
    data = snapshot()
    original = data["batches"][0]["documents"][0]
    conflicting = deepcopy(original)
    conflicting["document"].update(publication_year=2021, publication_date="2021-06-01")
    _, report = prepare([original, conflicting], OPTIONS)
    assert report["retained_studies"] == 0
    assert report["rejected"]["conflicting_year"] == 1


def test_likely_versions_merge_with_provenance_but_different_authors_do_not():
    doc = snapshot()["batches"][0]["documents"][0]
    second = deepcopy(doc)
    second["document_key"], second["revision_id"] = "doi:10.9999/version", "version"
    second["document"].update(doi="10.9999/version", publication_year=2021, publication_date="2021-06-01")
    studies, report = prepare([doc, second], OPTIONS)
    assert report["possible_versions_merged"] == 1
    assert len(studies) == 1 and studies[0]["year"] == 2020
    assert len(studies[0]["versions"]) == 2
    second["document"]["authors"] = ["Another author"]
    studies, _ = prepare([doc, second], OPTIONS)
    assert len(studies) == 2


def test_missing_text_or_year_is_not_fabricated():
    doc = snapshot()["batches"][0]["documents"][0]
    doc["document"]["abstract"] = None
    studies, report = prepare([doc], OPTIONS)
    assert studies == []
    assert report["rejected"] == {"missing_or_short_abstract": 1}
    doc["document"]["publication_year"] = None
    _, report = prepare([doc], OPTIONS)
    assert report["rejected"] == {"unknown_or_future_year": 1}


def test_incomplete_periods_cannot_produce_claim_of_comparable_growth():
    data = snapshot()
    data["history"]["periods"][-1]["job"]["source_exhausted"] = False
    result = analyze(unpack_snapshot(data), OPTIONS)
    assert not result["growth_data_comparable"]
    assert not result["candidates"]
    assert all(c["status"] in {"exploratory_candidate", "established"} for c in research_groups(result))


def test_zero_baseline_has_no_infinite_growth():
    values = metrics([{"year":2024}, {"year":2025}], {y:10 for y in range(2020,2026)}, list(range(2020,2026)), .5)
    assert values["growth_ratio"] == pytest.approx(5)
    assert values["raw_growth_ratio"] is None
    assert not values["growth_pattern"]
    json.dumps(values, allow_nan=False)


def test_snapshot_cannot_be_relabelled_as_another_direction():
    with pytest.raises(AnalysisInputError, match="другому направлению"):
        analyze(unpack_snapshot(snapshot()), OPTIONS.model_copy(update={"topic":"quantum computing"}))


def test_cancellation_stops_before_model_fit():
    event = Event()
    event.set()
    with pytest.raises(CancelledError):
        analyze(unpack_snapshot(snapshot()), OPTIONS, cancel=event)


def test_export_cannot_overwrite_the_input_even_with_overwrite_enabled(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text("original")
    with pytest.raises(AnalysisInputError, match="исходный корпус"):
        export_result({}, path, protected_paths=[path], overwrite=True)
    assert path.read_text(encoding="utf-8") == "original"
    output = tmp_path / "result.json"
    export_result({"candidates":[]}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"candidates":[]}


def test_bad_or_oversized_inputs_fail_explicitly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a corpus"}')
    with pytest.raises(AnalysisInputError):
        read_snapshot(path)
    with pytest.raises(AnalysisInputError, match="больше"):
        unpack_snapshot(snapshot(), max_documents=1)


@pytest.mark.parametrize("passage", [
    "Our device does not improve energy efficiency and does not reduce latency.",
    "Our design cannot improve the processing speed of this optical system.",
    "Our device fails to improve energy efficiency under these conditions.",
    "Our device doesn't improve energy efficiency under these conditions.",
    "An improvement in energy efficiency was not observed in our measurements.",
    "To accomplish neuromorphic computing, highly efficient optoelectronic synapses, "
    "which can be the building blocks of optoelectronic neuromorphic computers, are necessary.",
    "Efficient optical processing is required to make such devices practical.",
])
def test_advantage_rejects_negative_results_and_requirements(passage):
    study = {"id": "negative", "title": "Photonic neural device", "abstract": passage,
             "url": "https://example.org/negative"}
    assert evidence_card([study])["advantage"] is None


@pytest.mark.parametrize("passage", [
    "Our device improves energy efficiency and reduces latency under the tested conditions.",
    "Our device not only improves efficiency but also reduces latency.",
    "Our device reduces latency without requiring additional optical components.",
])
def test_advantage_preserves_supported_results_and_source(passage):
    study = {"id": "positive", "title": "Photonic neural device", "abstract": passage,
             "url": "https://example.org/positive"}
    card = evidence_card([study])["advantage"]
    assert card["text"] == passage
    assert card["study_id"] == study["id"]
    assert card["url"] == study["url"]


def test_advantage_uses_another_source_when_first_result_is_negative():
    negative = {"id": "negative", "title": "Photonic neural device",
                "abstract": "Our device does not improve energy efficiency and does not reduce latency.",
                "url": "https://example.org/negative"}
    positive = {"id": "positive", "title": "Another photonic neural device",
                "abstract": "Our device improves energy efficiency and reduces latency in measured experiments.",
                "url": "https://example.org/positive"}
    card = evidence_card([negative, positive])["advantage"]
    assert card["study_id"] == positive["id"]
    assert card["text"] == positive["abstract"]
    assert card["url"] == positive["url"]


def test_cancelled_history_keeps_saved_periods_available_for_analysis(tmp_path):
    started = Event()

    class PausingProvider(Provider):
        def iter_pages(self, request, cancel):
            if request.from_date.year == 2023:
                started.set()
                assert cancel.wait(10), "History cancellation timed out"
            yield from super().iter_pages(request, cancel)

    with Backend(BackendSettings(data_dir=tmp_path, history_period_delay_seconds=0),
                 provider_factory=PausingProvider) as backend:
        identifier = backend.submit_history(HistoryRequest(
            topic=OPTIONS.topic, sources=["openalex"], period="year", auto_split=False,
            from_date=date(2020, 1, 1), until_date=date(2025, 12, 31), max_results_per_period=100))
        assert started.wait(5)
        assert backend.cancel_history(identifier)
        history = backend.wait_history(identifier, timeout=5)
        assert history.state == "cancelled"
        assert any(p.job is None for p in history.periods)
        corpus = read_history(backend, identifier)
        assert len(corpus["entries"]) == 54
        result = analyze(corpus, OPTIONS)
        assert research_groups(result)
        assert result["coverage"]["2024"] and result["coverage"]["2025"]
        assert not result["growth_data_comparable"]
        assert not result["candidates"]
    assert all(c["status"] in {"exploratory_candidate", "established"} for c in research_groups(result))


@pytest.mark.parametrize("bad_job", ["missing", "wrong_request"])
def test_completed_history_still_rejects_missing_or_inconsistent_jobs(bad_job):
    data = snapshot()
    if bad_job == "missing":
        data["history"]["periods"][0]["job"] = None
    else:
        data["history"]["periods"][0]["job"]["request"]["topic"] = "unrelated direction"
    with pytest.raises(AnalysisInputError):
        unpack_snapshot(data)

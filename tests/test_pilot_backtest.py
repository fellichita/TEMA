"""As-of boundary regressions: future data cannot change strict predictions."""
from datetime import date, datetime, UTC

import pytest

from scripts.backtest_pilot import FrozenAsOfConfiguration, prepare_asof, replay_at_cutoff
from scripts.evaluate_pilot import EvaluationError
from tests.test_pilot_evidence import document


def configuration(**changes):
    return FrozenAsOfConfiguration.model_validate(dict(cutoff=date(2022, 12, 31), mode="strict",
        query_known_at=date(2022, 1, 1), aliases_known_at=date(2022, 1, 1),
        rules_frozen_at=date(2022, 1, 1), encoder_training_cutoff=date(2021, 1, 1),
        encoder_digest="a" * 64, rules_digest="b" * 64, query_digest="c" * 64) | changes)


def archived(index, year=2021, observed=2022, **changes):
    item = document(index, year=year)
    return type(item).model_validate(item.model_dump(mode="python") | dict(
        fetched_at=datetime(observed, 7, 1, tzinfo=UTC), date_precision="day", publication_date=date(year, 6, 1)) | changes)


def test_future_metadata_and_publications_cannot_change_strict_prediction():
    original = archived(1)
    future_revision = archived(1, observed=2025, title="Renamed technology with future terminology")
    future_paper = archived(2, year=2024, observed=2024)
    discover = lambda docs: [{"id": item.document_key, "title": item.title} for item in docs]
    before = replay_at_cutoff([original], configuration(), discover)
    after = replay_at_cutoff([original, future_revision, future_paper], configuration(), discover)
    assert before["prediction_hash"] == after["prediction_hash"]
    assert before["corpus_hash"] == after["corpus_hash"]
    assert after["excluded"] == {"unknown_availability": 0, "future_availability": 1, "late_observation": 1}
    assert after["precision"] is None and after["lead_time"] is None
    assert after["status"] == "predictions_replayed_outcomes_not_evaluated"


def test_unknown_public_availability_is_not_backdated_from_year():
    unknown = archived(1, publication_date=None, date_precision="year")
    prepared, report = prepare_asof([unknown], configuration())
    assert not prepared and report["excluded"]["unknown_availability"] == 1


def test_future_aliases_and_unknown_model_provenance_block_strict_claim():
    with pytest.raises(EvaluationError, match="future query or alias"):
        prepare_asof([], configuration(aliases_known_at=date(2026, 1, 1)))
    with pytest.raises(EvaluationError, match="encoder"):
        prepare_asof([], configuration(encoder_training_cutoff=None))
    with pytest.raises(EvaluationError, match="rules"):
        prepare_asof([], configuration(rules_frozen_at=date(2026, 1, 1)))


def test_reconstructed_mode_strips_current_citations_and_labels_limitations():
    item = archived(1, observed=2026, raw_metadata={"cited_by_count": 999, "counts_by_year": [{"year": 2025}],
        "title": "real preserved metadata", "is_retracted": True})
    prepared, report = prepare_asof([item], configuration(mode="reconstructed", encoder_training_cutoff=None))
    assert len(prepared) == 1
    assert "cited_by_count" not in prepared[0].raw_metadata
    assert "counts_by_year" not in prepared[0].raw_metadata
    assert prepared[0].raw_metadata["title"] == "real preserved metadata"
    assert any("reconstructed" in item for item in report["limitations"])
    assert report["status"] == "corpus_prepared_not_scientifically_evaluated"

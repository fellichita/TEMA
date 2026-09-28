"""Collection defects remain blocking unless their exact soft cause is proven."""

from copy import deepcopy
import hashlib
import json

import pytest

from app.ml.corpus import read_snapshot, unpack_snapshot
from tests.mvp_fixture import snapshot


INCOMPLETE = "Сбор периода неполный"
SKIPPED = "Выдача не исчерпана или есть пропуски"
EXPORT = "Неполный экспорт периода"
OUTSIDE = "Документы вне периода запроса"
BAD_COUNTS = "Несогласованные счётчики сбора периода"
UNKNOWN_COUNTS = "Недостаточно данных о счётчиках сбора периода"
UNKNOWN_YEAR = "Год публикации неизвестен; принадлежность периоду не подтверждена."


@pytest.fixture
def collected_snapshot():
    data = snapshot()
    data["history"]["periods"] = data["history"]["periods"][:1]
    data["batches"] = data["batches"][:1]
    batch = data["batches"][0]
    batch["documents"] = batch["documents"][:2]
    batch["total"] = 2
    period = data["history"]["periods"][0]
    period.update(state="partial", incomplete_reason="invalid_records_skipped")
    period["job"].update(stored=2, skipped=1, scanned=3, total_available=3)
    return data


def unpacked_period(data):
    return unpack_snapshot(data)["periods"][0]


def test_exhausted_invalid_records_are_quality_notes_without_erasing_issues(collected_snapshot):
    original = deepcopy(collected_snapshot)
    period = unpacked_period(collected_snapshot)
    assert period["issues"] == sorted([INCOMPLETE, SKIPPED])
    assert period["nonblocking_issues"] == sorted([INCOMPLETE, SKIPPED])
    assert len(period["data_quality_notes"]) == 1
    note = period["data_quality_notes"][0]
    for text in ("skipped=1", "scanned=3", "stored=2", "export=2", "total_available=3"):
        assert text in note
    assert collected_snapshot == original


@pytest.mark.parametrize("duplicates", [1, 5, 100])
def test_backend_identity_deduplication_is_not_an_export_gap(collected_snapshot, duplicates):
    job = collected_snapshot["history"]["periods"][0]["job"]
    job.update(scanned=3 + duplicates, total_available=3 + duplicates)
    period = unpacked_period(collected_snapshot)
    assert period["nonblocking_issues"] == sorted([INCOMPLETE, SKIPPED])
    assert EXPORT not in period["issues"]


@pytest.mark.parametrize("total", [None, 2, 3])
def test_explicit_exhaustion_does_not_require_a_known_exact_api_total(collected_snapshot, total):
    collected_snapshot["history"]["periods"][0]["job"]["total_available"] = total
    period = unpacked_period(collected_snapshot)
    assert period["nonblocking_issues"] == sorted([INCOMPLETE, SKIPPED])


@pytest.mark.parametrize(("location", "field", "value"), [
    ("period", "state", "complete"),
    ("period", "state", "cancelled"),
    ("period", "state", "interrupted"),
    ("period", "state", "failed"),
    ("period", "incomplete_reason", None),
    ("period", "incomplete_reason", "source_not_exhausted"),
    ("period", "incomplete_reason", "inconsistent_total"),
    ("period", "incomplete_reason", "daily_limit_reached"),
    ("period", "incomplete_reason", "period_budget_exceeded"),
    ("period", "incomplete_reason", "unrecognized_reason"),
    ("job", "state", "failed"),
    ("job", "state", "cancelled"),
    ("job", "state", "interrupted"),
    ("job", "state", "running"),
    ("job", "source_exhausted", False),
    ("job", "skipped", 0),
    ("job", "skipped", 2),
    ("job", "scanned", 0),
    ("job", "scanned", 2),
    ("job", "stored", 1),
    ("job", "stored", 3),
    ("job", "total_available", 4),
    ("batch", "total", 1),
    ("batch", "total", 3),
])
def test_each_failed_proof_condition_keeps_the_period_blocking(collected_snapshot, location, field, value):
    period = collected_snapshot["history"]["periods"][0]
    target = {"period": period, "job": period["job"], "batch": collected_snapshot["batches"][0]}[location]
    target[field] = value
    result = unpacked_period(collected_snapshot)
    assert result["issues"]
    assert result["nonblocking_issues"] == []
    assert result["data_quality_notes"] == []


@pytest.mark.parametrize(("location", "field"), [
    ("period", "incomplete_reason"), ("job", "state"), ("job", "source_exhausted"),
    ("job", "scanned"), ("job", "stored"), ("job", "skipped"), ("batch", "total"),
])
def test_missing_legacy_proof_never_turns_into_a_default_success(collected_snapshot, location, field):
    period = collected_snapshot["history"]["periods"][0]
    target = {"period": period, "job": period["job"], "batch": collected_snapshot["batches"][0]}[location]
    target.pop(field)
    result = unpacked_period(collected_snapshot)
    assert result["issues"]
    assert result["nonblocking_issues"] == []
    assert result["data_quality_notes"] == []


def test_missing_batch_remains_blocking(collected_snapshot):
    collected_snapshot["batches"] = []
    period = unpacked_period(collected_snapshot)
    assert EXPORT in period["issues"]
    assert period["nonblocking_issues"] == []
    assert period["data_quality_notes"] == []


def test_outside_publication_is_never_softened_by_skipped_record_proof(collected_snapshot):
    document = collected_snapshot["batches"][0]["documents"][0]["document"]
    document.update(publication_year=2021, publication_date="2021-06-01")
    period = unpacked_period(collected_snapshot)
    assert OUTSIDE in period["issues"]
    assert OUTSIDE not in period["nonblocking_issues"]
    assert set(period["issues"]) - set(period["nonblocking_issues"]) == {OUTSIDE}


def test_complete_clean_collection_has_no_new_warnings(collected_snapshot):
    period = collected_snapshot["history"]["periods"][0]
    period.update(state="complete", incomplete_reason=None)
    period["job"].update(skipped=0, scanned=2, total_available=2)
    result = unpacked_period(collected_snapshot)
    assert result["issues"] == []
    assert result["nonblocking_issues"] == []
    assert result["data_quality_notes"] == []


def test_absent_job_has_explicit_empty_quality_fields(collected_snapshot):
    collected_snapshot["history"]["periods"][0]["job"] = None
    period = unpacked_period(collected_snapshot)
    assert INCOMPLETE in period["issues"]
    assert period["nonblocking_issues"] == []
    assert period["data_quality_notes"] == []


def test_source_json_and_backend_status_are_preserved(collected_snapshot, tmp_path):
    path = tmp_path / "source.json"
    original = json.dumps(collected_snapshot, ensure_ascii=False).encode()
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    corpus = read_snapshot(path)
    assert path.read_bytes() == original
    assert corpus["provenance"]["snapshot_sha256"] == digest
    raw = json.loads(path.read_bytes())
    assert raw["history"]["periods"][0]["state"] == "partial"
    assert raw["history"]["periods"][0]["incomplete_reason"] == "invalid_records_skipped"
    assert corpus["periods"][0]["nonblocking_issues"]


@pytest.mark.parametrize(("stored", "skipped", "scanned"), [(2, 0, 1), (2, 1, 2), (2, 3, 1)])
def test_complete_state_cannot_override_impossible_collection_counts(collected_snapshot, stored, skipped, scanned):
    period = collected_snapshot["history"]["periods"][0]
    period.update(state="complete", incomplete_reason=None)
    period["job"].update(stored=stored, skipped=skipped, scanned=scanned, total_available=None)
    result = unpacked_period(collected_snapshot)
    assert BAD_COUNTS in result["issues"]
    assert BAD_COUNTS not in result["nonblocking_issues"]


@pytest.mark.parametrize("missing", ["scanned", "stored", "skipped"])
def test_complete_state_with_unknown_required_count_is_blocking(collected_snapshot, missing):
    period = collected_snapshot["history"]["periods"][0]
    period.update(state="complete", incomplete_reason=None)
    period["job"].update(stored=2, skipped=0, scanned=2, total_available=None)
    period["job"].pop(missing)
    result = unpacked_period(collected_snapshot)
    assert UNKNOWN_COUNTS in result["issues"]
    assert UNKNOWN_COUNTS not in result["nonblocking_issues"]


@pytest.mark.parametrize("duplicates", [0, 1, 5])
@pytest.mark.parametrize("known_total", [False, True])
def test_valid_complete_collection_with_deduplication_and_unknown_api_total_stays_clean(
        collected_snapshot, duplicates, known_total):
    period = collected_snapshot["history"]["periods"][0]
    period.update(state="complete", incomplete_reason=None)
    period["job"].update(stored=2, skipped=0, scanned=2 + duplicates,
                         total_available=2 + duplicates if known_total else None)
    result = unpacked_period(collected_snapshot)
    assert result["issues"] == []
    assert result["nonblocking_issues"] == []


@pytest.mark.parametrize("unknown_count", [1, 2])
@pytest.mark.parametrize("state", ["complete", "partial"])
def test_unknown_year_is_a_local_hard_issue_despite_other_usable_publications(
        collected_snapshot, unknown_count, state):
    period = collected_snapshot["history"]["periods"][0]
    if state == "complete":
        period.update(state="complete", incomplete_reason=None)
        period["job"].update(skipped=0, scanned=2, total_available=2)
    for entry in collected_snapshot["batches"][0]["documents"][:unknown_count]:
        entry["document"].update(publication_year=None, publication_date=None,
                                 publication_month=None, date_precision="unknown")
    original = deepcopy(collected_snapshot)
    result = unpacked_period(collected_snapshot)
    assert result["undated_records"] == unknown_count
    assert result["issues"].count(UNKNOWN_YEAR) == 1
    assert UNKNOWN_YEAR not in result["nonblocking_issues"]
    assert collected_snapshot == original


def test_known_dates_and_missing_job_have_zero_undated_record_counts(collected_snapshot):
    assert unpacked_period(collected_snapshot)["undated_records"] == 0
    collected_snapshot["history"]["periods"][0]["job"] = None
    assert unpacked_period(collected_snapshot)["undated_records"] == 0


def test_unknown_year_before_scoring_window_only_blocks_its_own_period():
    from app.ml.engine import growth_assessment

    data = snapshot()
    old_period = deepcopy(data["history"]["periods"][0])
    old_period.update(id="2019", from_date="2019-01-01", until_date="2019-12-31")
    old_period["job"].update(id="2019", scanned=1, stored=1, skipped=0, total_available=1)
    old_period["job"]["request"].update(from_date="2019-01-01", until_date="2019-12-31")
    old_entry = deepcopy(data["batches"][0]["documents"][0])
    old_entry["document"].update(publication_year=None, publication_date=None,
                                  publication_month=None, date_precision="unknown")
    data["history"]["periods"].insert(0, old_period)
    data["batches"].insert(0, {"job_id": "2019", "total": 1, "documents": [old_entry]})
    corpus = unpack_snapshot(data)
    assert corpus["periods"][0]["undated_records"] == 1
    assert UNKNOWN_YEAR in corpus["periods"][0]["issues"]
    years = list(range(2020, 2026))
    current = growth_assessment(corpus["periods"], years, dict.fromkeys(years, 10))
    assert current["growth_data_comparable"]
    assert not any(current["growth_comparability"]["blocking_issues"].values())
    past = growth_assessment(corpus["periods"], [2019], {2019: 10})
    assert not past["growth_data_comparable"]
    assert UNKNOWN_YEAR in past["growth_comparability"]["blocking_issues"]["2019"]

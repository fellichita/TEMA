"""Protect audit conclusions against duplicate versions and incomplete exports."""

from copy import deepcopy

import pytest

from scripts.audit_ml_input import audit, full_year_covered


def entry(key="doi:10.1234/a", year=2020, **changes):
    document = {"source": "openalex", "source_id": "W1", "doi": "10.1234/a",
                "title": "Optical neural computing", "abstract": "An optical neural computation experiment.",
                "publication_year": year, "publication_date": f"{year}-06-01" if year else None,
                "document_type": "article", "language": "en", "url": "https://doi.org/10.1234/a",
                "fetched_at": "2026-09-06T00:00:00Z"}
    document.update(changes)
    return {"document_key": key, "revision_id": f"revision-{year}-{key}", "document": document}


def snapshot(*year_entries):
    periods, batches = [], []
    for year, entries in year_entries:
        request = {"topic": "photonic neuromorphic computing", "source": "openalex",
                   "from_date": f"{year}-01-01", "until_date": f"{year}-12-31"}
        job = {"id": f"job-{year}", "request": request, "state": "succeeded", "scanned": len(entries),
               "stored": len(entries), "skipped": 0, "total_available": len(entries), "source_exhausted": True}
        periods.append({"id": f"period-{year}", "source": "openalex", "state": "complete", "job": job,
                        "from_date": request["from_date"], "until_date": request["until_date"]})
        batches.append({"job_id": job["id"], "total": len(entries), "documents": entries})
    return {"schema_version": 1, "history": {"id": "test-history", "contract_version": 1,
             "request": {"topic": "photonic neuromorphic computing"}, "periods": periods,
             "state": "succeeded", "coverage_complete": True}, "batches": batches}


def test_conflicting_years_are_quarantined_instead_of_using_latest_year():
    data = snapshot((2020, [entry()]), (2021, [entry(year=2021, source_id="W2")]))
    report, sample = audit(data, start_year=2020, end_year=2021)
    assert report["all_exported_occurrences"] == 2
    assert report["backend_identities"] == 1
    assert report["focus_stable_fields"]["count"] == 0
    assert report["conflict_counts"]["publication_year"] == 1
    assert all("identity_or_date_conflicts" in row["coverage_issues"] for row in report["years"])
    assert sample == []


def test_repeat_is_not_counted_as_two_studies_and_input_is_unchanged():
    data = snapshot((2020, [entry(), entry()]))
    before = deepcopy(data)
    report, sample = audit(data, start_year=2020, end_year=2020)
    assert report["extra_occurrences"] == 1
    assert report["focus_stable_fields"]["count"] == 1
    assert len(sample) == 1
    assert data == before


def test_missing_year_is_not_filled_from_collection_period():
    report, sample = audit(snapshot((2020, [entry(year=None)])), start_year=2020, end_year=2020)
    assert report["all_identity_fields_latest_version"]["missing_or_invalid_year"] == 1
    assert report["focus_stable_fields"]["count"] == 0
    assert "unknown_document_year" in report["years"][0]["coverage_issues"]
    assert sample == []


def test_backend_complete_does_not_hide_export_truncation_or_absent_year():
    data = snapshot((2020, [entry()]))
    data["batches"][0]["total"] = 2
    report, _ = audit(data, start_year=2020, end_year=2021)
    assert "export_count_mismatch" in report["years"][0]["coverage_issues"]
    assert "calendar_gap" in report["years"][1]["coverage_issues"]


def test_same_doi_under_different_keys_is_flagged_even_with_url_prefix():
    data = snapshot((2020, [entry(), entry(key="other-key", source_id="W2", doi="https://doi.org/10.1234/A")]))
    report, _ = audit(data, start_year=2020, end_year=2020)
    assert report["cross_key_alias_collisions"] == [{"alias": "doi:10.1234/a",
                                                    "document_keys": ["doi:10.1234/a", "other-key"]}]
    assert report["focus_stable_fields"]["count"] == 0


def test_same_title_is_only_a_possible_duplicate_and_missing_abstract_stays_missing():
    data = snapshot((2020, [entry(), entry(key="doi:10.1234/b", doi="10.1234/b", source_id="W2", abstract=None)]))
    report, _ = audit(data, start_year=2020, end_year=2020)
    assert report["same_normalized_title_groups"] == 1
    assert report["focus_stable_fields"]["count"] == 2
    assert report["focus_stable_fields"]["missing_abstract"] == 1


def test_missing_month_is_not_a_complete_calendar_year():
    periods = [{"from_date": "2020-01-01", "until_date": "2020-05-31"},
               {"from_date": "2020-07-01", "until_date": "2020-12-31"}]
    assert not full_year_covered(periods, 2020)
    periods.append({"from_date": "2020-06-01", "until_date": "2020-06-30"})
    assert full_year_covered(periods, 2020)


def test_sample_does_not_depend_on_batch_document_order():
    entries = [entry(key=f"key-{i}", doi=f"10.1234/{i}", source_id=f"W{i}") for i in range(10)]
    _, forward = audit(snapshot((2020, entries)), start_year=2020, end_year=2020)
    _, backward = audit(snapshot((2020, entries[::-1])), start_year=2020, end_year=2020)
    assert forward == backward
    assert len(forward) == 4


def test_duplicate_job_batches_fail_instead_of_silently_replacing_data():
    data = snapshot((2020, [entry()]))
    data["batches"].append(deepcopy(data["batches"][0]))
    with pytest.raises(ValueError, match="Duplicate batch"):
        audit(data)

"""External snapshot validation and publication intervals without source rewriting."""

from copy import deepcopy
import hashlib
import json

import pytest

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import read_snapshot, unpack_snapshot
from app.ml.engine import analyze
from tests.mvp_fixture import snapshot


OUTSIDE_PERIOD = "Документы вне периода запроса"
OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


@pytest.fixture
def small_snapshot():
    data = snapshot()
    data["history"]["periods"] = data["history"]["periods"][:1]
    data["batches"] = data["batches"][:1]
    data["batches"][0]["documents"] = data["batches"][0]["documents"][:1]
    data["batches"][0]["total"] = 1
    data["history"]["periods"][0]["job"].update(stored=1, scanned=1, total_available=1)
    return data


def document(data):
    return data["batches"][0]["documents"][0]["document"]


def set_period(data, start, end):
    period = data["history"]["periods"][0]
    period.update(from_date=start, until_date=end)
    period["job"]["request"].update(from_date=start, until_date=end)


@pytest.mark.parametrize("field", ["revision_id", "document_key"])
@pytest.mark.parametrize("value", [None, 42, [], "", "  ", "missing"])
def test_bad_required_identity_fails_during_json_import(small_snapshot, tmp_path, field, value):
    entry = small_snapshot["batches"][0]["documents"][0]
    if value == "missing":
        del entry[field]
    else:
        entry[field] = value
    path = tmp_path / "bad-identity.json"
    payload = json.dumps(small_snapshot).encode()
    path.write_bytes(payload)
    with pytest.raises(AnalysisInputError):
        read_snapshot(path)
    assert path.read_bytes() == payload


@pytest.mark.parametrize(("field", "value"), [
    ("abstract", 123), ("abstract", []), ("abstract", False),
    ("doi", 123), ("doi", []), ("doi", "not-a-doi"), ("doi", ""), ("title", None),
    ("authors", "A. Researcher"), ("authors", [42]), ("authors", [None]),
    ("source_id", 123), ("url", 123), ("document_type", []),
    ("fetched_at", 123), ("language", []),
    ("raw_metadata", []), ("raw_metadata", "retracted"),
    ("publication_year", "2020"), ("publication_year", 2020.0),
    ("publication_year", True), ("publication_year", 0), ("publication_year", 10000),
    ("publication_month", "6"), ("publication_month", True),
    ("publication_month", 0), ("publication_month", 13),
    ("publication_date", ""), ("publication_date", 0), ("publication_date", "2020-02-30"),
    ("date_precision", []), ("date_precision", "decade"),
])
def test_bad_consumed_document_fields_fail_during_json_import(small_snapshot, tmp_path, field, value):
    document(small_snapshot)[field] = value
    original = deepcopy(small_snapshot)
    path = tmp_path / "bad-document.json"
    path.write_text(json.dumps(small_snapshot), encoding="utf-8")
    with pytest.raises(AnalysisInputError):
        read_snapshot(path)
    assert small_snapshot == original


@pytest.mark.parametrize(("path", "value"), [
    (("schema_version",), True), (("schema_version",), 1.0),
    (("history", "contract_version"), True), (("history", "contract_version"), 2.0),
    (("history", "id"), 123), (("history", "state"), None),
    (("history", "request", "topic"), 123),
    (("history", "request", "primary_topic_ids"), "T1"),
    (("history", "periods"), {}), (("batches",), {}),
    (("history", "periods", 0, "state"), []),
    (("history", "periods", 0, "from_date"), "2020-02-30"),
    (("history", "periods", 0, "until_date"), None),
    (("history", "periods", 0, "job", "id"), 2020),
    (("history", "periods", 0, "job", "source_exhausted"), "false"),
    (("history", "periods", 0, "job", "source_exhausted"), 1),
    (("history", "periods", 0, "job", "stored"), 1.0),
    (("history", "periods", 0, "job", "scanned"), "1"),
    (("history", "periods", 0, "job", "skipped"), False),
    (("history", "periods", 0, "job", "total_available"), -1),
    (("batches", 0, "total"), True), (("batches", 0, "documents"), {}),
])
def test_bad_snapshot_structure_fails_at_import(small_snapshot, path, value):
    target = small_snapshot
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AnalysisInputError):
        unpack_snapshot(small_snapshot)


@pytest.mark.parametrize("payload", [None, [], 1, "snapshot"])
def test_non_object_json_fails_as_analysis_input_error(payload):
    with pytest.raises(AnalysisInputError):
        unpack_snapshot(payload)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e309", "-1e309"])
@pytest.mark.parametrize("location", ["consumed_field", "raw_metadata", "provenance", "unused_field"])
def test_nonfinite_json_numbers_fail_during_decode(small_snapshot, tmp_path, literal, location):
    marker = "nonfinite-json-test-value"
    if location == "consumed_field":
        document(small_snapshot)["abstract"] = marker
    elif location == "raw_metadata":
        document(small_snapshot)["raw_metadata"] = {"provider_metric": [marker]}
    elif location == "provenance":
        small_snapshot["history"]["request"]["provider_extension"] = {"metric": marker}
    else:
        small_snapshot["unconsumed_export_metadata"] = {"metric": marker}
    payload = json.dumps(small_snapshot).replace(json.dumps(marker), literal).encode()
    path = tmp_path / "nonfinite.json"
    path.write_bytes(payload)
    with pytest.raises(AnalysisInputError, match="JSON"):
        read_snapshot(path)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("location", ["raw_metadata", "provenance", "unused_field"])
def test_finite_json_numbers_and_nonfinite_names_as_text_are_preserved(small_snapshot, tmp_path, location):
    metadata = {"metrics": [1e308, -1e308, 1e-309, 0.5],
                "labels": ["NaN", "Infinity", "-Infinity", "1e309"]}
    if location == "raw_metadata":
        document(small_snapshot)["raw_metadata"] = metadata
    elif location == "provenance":
        small_snapshot["history"]["request"]["provider_extension"] = metadata
    else:
        small_snapshot["unconsumed_export_metadata"] = metadata
    payload = json.dumps(small_snapshot, allow_nan=False).encode()
    path = tmp_path / "finite.json"
    path.write_bytes(payload)
    corpus = read_snapshot(path)
    assert corpus["entries"] == small_snapshot["batches"][0]["documents"]
    assert corpus["provenance"]["request"] == small_snapshot["history"]["request"]
    json.dumps(corpus, allow_nan=False)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("flag", ["true", "false", 1, 0, []])
def test_malformed_retraction_flag_fails_at_import(small_snapshot, flag):
    document(small_snapshot)["raw_metadata"] = {"is_retracted": flag}
    with pytest.raises(AnalysisInputError):
        unpack_snapshot(small_snapshot)


@pytest.mark.parametrize("flag", [True, False, None])
def test_boolean_and_unknown_retraction_flags_remain_unchanged(small_snapshot, flag):
    metadata = {"is_retracted": flag, "unconsumed_provider_field": [42, {"arbitrary": True}]}
    document(small_snapshot)["raw_metadata"] = metadata
    corpus = unpack_snapshot(small_snapshot)
    assert corpus["entries"][0]["document"]["raw_metadata"] == metadata


@pytest.mark.parametrize("doi", ["10.9999/MIXED", "doi:10.9999/MIXED",
                                 "http://doi.org/10.9999/MIXED", "https://doi.org/10.9999/MIXED",
                                 "https://dx.doi.org/10.9999/MIXED", "http://dx.doi.org/10.9999/MIXED",
                                 "  HTTP://DX.DOI.ORG/10.9999/MIXED  "])
def test_valid_doi_aliases_are_validated_without_normalizing_input(small_snapshot, doi):
    document(small_snapshot)["doi"] = doi
    corpus = unpack_snapshot(small_snapshot)
    assert corpus["entries"][0]["document"]["doi"] == doi
    assert document(small_snapshot)["doi"] == doi


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("source", ["openalex", "crossref"])
def test_valid_snapshots_preserve_exact_records_and_provenance(small_snapshot, tmp_path, version, source):
    small_snapshot["history"]["contract_version"] = version
    small_snapshot["history"]["request"]["sources"] = [source]
    period = small_snapshot["history"]["periods"][0]
    period["source"] = period["job"]["request"]["source"] = source
    document(small_snapshot).update(source=source, doi="https://doi.org/10.9999/MIXED",
                                    title="  Optical neural computing experiment  ")
    original = deepcopy(small_snapshot)
    payload = json.dumps(small_snapshot, ensure_ascii=False).encode()
    path = tmp_path / "valid.json"
    path.write_bytes(payload)
    corpus = read_snapshot(path)
    assert corpus["entries"] == original["batches"][0]["documents"]
    assert corpus["provenance"]["request"] == original["history"]["request"]
    assert corpus["provenance"]["snapshot_sha256"] == hashlib.sha256(payload).hexdigest()
    assert corpus["periods"][0]["issues"] == []
    assert small_snapshot == original
    assert path.read_bytes() == payload


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_minimal_documents_and_missing_optional_fields_remain_accepted(small_snapshot, version):
    small_snapshot["history"]["contract_version"] = version
    doc = document(small_snapshot)
    for key in ("authors", "fetched_at", "raw_metadata", "date_precision", "publication_month", "language"):
        del doc[key]
    doc.update(abstract=None, doi=None)
    original = deepcopy(small_snapshot)
    corpus = unpack_snapshot(small_snapshot)
    assert corpus["entries"] == original["batches"][0]["documents"]
    assert corpus["periods"][0]["issues"] == []
    assert small_snapshot == original


@pytest.mark.parametrize(("date_fields", "start", "end", "outside"), [
    ({"publication_year": 2024, "date_precision": "year"}, "2020-01-01", "2020-12-31", True),
    ({"publication_year": 2019, "date_precision": "year"}, "2020-01-01", "2020-12-31", True),
    ({"publication_year": 2020, "date_precision": "year"}, "2020-06-15", "2020-06-20", False),
    ({"publication_year": 2020, "date_precision": "year"}, "2019-12-31", "2020-01-01", False),
    ({"publication_year": 2020, "publication_month": 5, "date_precision": "month"},
     "2020-06-01", "2020-06-30", True),
    ({"publication_year": 2020, "publication_month": 7, "date_precision": "month"},
     "2020-06-01", "2020-06-30", True),
    ({"publication_year": 2020, "publication_month": 6, "date_precision": "month"},
     "2020-06-15", "2020-06-20", False),
    ({"publication_year": 2020, "publication_month": 2, "date_precision": "month"},
     "2020-02-29", "2020-03-01", False),
    ({"publication_year": 2020, "publication_month": 2, "date_precision": "month"},
     "2020-03-01", "2020-03-02", True),
    ({"publication_year": 2020, "publication_date": "2020-06-01", "date_precision": "day"},
     "2020-06-02", "2020-06-30", True),
    ({"publication_year": 2020, "publication_date": "2020-06-01", "date_precision": "day"},
     "2020-06-01", "2020-06-01", False),
    ({"publication_year": 2024}, "2020-01-01", "2020-12-31", True),
    ({"publication_year": 2020}, "2020-06-15", "2020-06-20", False),
    ({"publication_year": None, "date_precision": "unknown"}, "2020-01-01", "2020-12-31", False),
])
def test_publication_precision_flags_only_disjoint_intervals(small_snapshot, date_fields, start, end, outside):
    doc = document(small_snapshot)
    for key in ("publication_year", "publication_month", "publication_date", "date_precision"):
        doc.pop(key, None)
    doc.update(date_fields)
    set_period(small_snapshot, start, end)
    original = deepcopy(small_snapshot)
    corpus = unpack_snapshot(small_snapshot)
    assert (OUTSIDE_PERIOD in corpus["periods"][0]["issues"]) is outside
    assert corpus["entries"] == original["batches"][0]["documents"]
    assert small_snapshot == original


def test_year_only_outside_period_reaches_existing_coverage_policy():
    data = snapshot()
    # Keep publication-period validation independent of possible-version merging.
    for batch in data["batches"]:
        for index, entry in enumerate(batch["documents"]):
            entry["document"]["authors"] = [f"Researcher Unique{batch['job_id']}_{index}"]
    assert analyze(unpack_snapshot(data), OPTIONS)["growth_data_comparable"]
    document(data).update(publication_year=2024, publication_date=None,
                          publication_month=None, date_precision="year")
    original = deepcopy(data)
    corpus = unpack_snapshot(data)
    assert OUTSIDE_PERIOD in corpus["periods"][0]["issues"]
    result = analyze(corpus, OPTIONS)
    assert OUTSIDE_PERIOD in result["coverage"]["2020"]
    assert not result["growth_data_comparable"]
    assert not result["candidates"]
    assert data == original


def test_valid_incomplete_counter_values_keep_coverage_issues(small_snapshot):
    job = small_snapshot["history"]["periods"][0]["job"]
    job.update(source_exhausted=False, total_available=None)
    corpus = unpack_snapshot(small_snapshot)
    assert "Выдача не исчерпана или есть пропуски" in corpus["periods"][0]["issues"]

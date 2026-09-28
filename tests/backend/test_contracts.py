from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage, normalize_doi, utc_now


def document(**changes):
    return DocumentRecord.model_validate({
        "source": "crossref", "source_id": "10.1234/test", "doi": "10.1234/test",
        "title": "Тестовая публикация", "url": "https://doi.org/10.1234/test", **changes,
    })


@pytest.mark.parametrize("values", [
    {"topic": " "}, {"topic": "AI\nresearch"}, {"topic": "AI", "max_results": 0},
    {"topic": "AI", "max_results": True}, {"topic": "AI", "max_results": 10001},
    {"topic": "AI", "from_date": "2025-01-01", "until_date": "2024-01-01"},
    {"topic": "AI", "source": "https://localhost"},
    {"topic": "AI", "unexpected": 1},
])
def test_invalid_requests(values):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(values)


def test_local_today_is_accepted_east_of_utc_but_tomorrow_is_still_the_future():
    """The first hours of a day east of UTC ran the local date ahead of the UTC one.

    Every analysis then failed on its first source request, because the window
    ends on the plan's local `as_of`. Both readings of "today" are accepted; the
    day after the later one is still refused.
    """
    latest = max(utc_now().date(), date.today())
    assert SearchRequest.model_validate({"topic": "AI", "until_date": date.today()}).until_date == date.today()
    assert SearchRequest.model_validate({"topic": "AI", "until_date": latest}).until_date == latest
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"topic": "AI", "until_date": latest + timedelta(days=1)})


def test_doi_normalization():
    assert normalize_doi(" https://doi.org/10.1234/ABC ") == "10.1234/abc"
    assert document(doi="DOI:10.1234/ABC").document_key == "doi:10.1234/abc"


@pytest.mark.parametrize("changes", [
    {"url": "file:///C:/secret"}, {"url": "https://name:password@example.com"},
    {"doi": "not-a-doi"}, {"publication_year": 2024},
    {"publication_year": 2024, "date_precision": "day"},
    {"publication_year": 2024, "date_precision": "month"},
    {"publication_year": 2024, "date_precision": "month", "publication_month": 13},
    {"publication_year": 2024, "date_precision": "year", "publication_date": "2024-01-01"},
    {"fetched_at": datetime(2024, 1, 1)},
])
def test_document_rejects_unsafe_or_invented_data(changes):
    with pytest.raises(ValidationError):
        document(**changes)


def test_source_counts_cannot_silently_disagree():
    with pytest.raises(ValidationError):
        SourcePage(documents=(document(),), scanned=2)


def test_year_precision_does_not_invent_a_day():
    result = document(publication_year=2024, date_precision="year")
    assert result.publication_date is None
    assert result.abstract is None


def test_primary_topic_ids_are_canonical_and_stable():
    request = SearchRequest(topic="Artificial intelligence", source="openalex",
                            primary_topic_ids=("T2", "https://openalex.org/T1"))
    assert request.primary_topic_ids == ("https://openalex.org/T1", "https://openalex.org/T2")
    assert SearchRequest.model_validate_json(request.model_dump_json()) == request
    assert SearchRequest(topic="legacy request").primary_topic_ids == ()


@pytest.mark.parametrize("changes", [
    {"source": "crossref"}, {"source": "epo"},
    {"primary_topic_ids": ["T1", "https://openalex.org/T1"]},
    {"primary_topic_ids": ["T1|T2"]}, {"primary_topic_ids": ["T1,publication_year:2025"]},
    {"primary_topic_ids": ["https://evil.test/T1"]}, {"primary_topic_ids": ["W1"]},
    {"primary_topic_ids": ["T0"]}, {"primary_topic_ids": ["T01"]},
    {"primary_topic_ids": [f"T{i}" for i in range(1, 102)]},
])
def test_primary_topics_reject_ambiguous_or_unsupported_scope(changes):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(dict(topic="Artificial intelligence", source="openalex",
                                          primary_topic_ids=["T1"]) | changes)

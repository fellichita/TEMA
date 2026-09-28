"""Bounded real provider adapters and untrusted coordinated Atom imports."""

from datetime import date
from threading import Event

import httpx
import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.enrichment import EnrichmentError, collect_patent_signal, import_arxiv_atom
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled
from tests.backend.test_epo import envelope, patent, provider
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen


@pytest.fixture
def patents(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    discovery = snapshot((document(1),), archive)
    return archive, frozen(candidate(discovery)), Context(), CredentialStore()


def lithium_patent(**kwargs):
    return patent(**kwargs).replace("Neuromorphic systems", "Lithium selective membranes")


def test_real_epo_oauth_and_xml_adapter_deduplicates_families_and_uses_public_dates(patents):
    archive, item, context, credentials = patents
    xml = envelope([lithium_patent(day="20240229"), lithium_patent(kind="B1", day="20250601"),
                    lithium_patent(number="2234567").replace('family-id="123"', '')])
    result = collect_patent_signal(item, query_plan(), archive, credentials, context,
        provider_factory=lambda: provider(lambda request: httpx.Response(200, text=xml)))
    assert len(result.documents) == 3 and len(result.families) == 1
    assert result.families[0].first_observed_publication_date == date(2024, 2, 29)
    assert result.families[0].first_observed_publication_year != 1999
    assert len(result.families[0].publications) == 2 and result.unresolved_family_publications == 1
    assert result.coverage.state == "complete" and not result.coverage.comparable
    assert all(archive.get(ref.revision_id).document_type == "patent" for ref in result.documents)
    assert "TEST_SECRET" not in result.model_dump_json()


def test_missing_keys_mean_unavailable_never_zero_interest(patents, monkeypatch):
    archive, item, context, credentials = patents
    monkeypatch.setattr(credentials, "get", lambda name: None)
    result = collect_patent_signal(item, query_plan(), archive, credentials, context)
    assert result.coverage.state == "unavailable" and "missing_epo_credentials" in result.coverage.reasons
    assert not result.documents and any("не доказывает" in value for value in result.limitations)


def test_empty_complete_query_is_still_only_query_coverage(patents):
    archive, item, context, credentials = patents
    result = collect_patent_signal(item, query_plan(), archive, credentials, context,
        provider_factory=lambda: provider(lambda request: httpx.Response(200, text=envelope([]))))
    assert result.coverage.state == "complete" and not result.families
    assert any("не охватывает" in value for value in result.limitations)


def test_patent_caps_api_errors_and_missing_dates_remain_partial(patents):
    archive, item, context, credentials = patents
    result = collect_patent_signal(item, query_plan(), archive, credentials, context, max_documents=1,
        provider_factory=lambda: provider(lambda request: httpx.Response(200, text=envelope([lithium_patent()], total=50))))
    assert len(result.documents) == 1 and result.coverage.state == "partial" and result.coverage.limit_reached
    unavailable = collect_patent_signal(item, query_plan(), archive, credentials, Context(),
        provider_factory=lambda: provider(lambda request: httpx.Response(429)))
    assert unavailable.coverage.state == "unavailable"
    unresolved = collect_patent_signal(item, query_plan(), archive, credentials, Context(),
        provider_factory=lambda: provider(lambda request: httpx.Response(200, text=envelope([lithium_patent(day="")]))))
    assert not unresolved.documents and unresolved.coverage.unresolved_records == 1


def test_offscope_title_and_future_publication_are_excluded(patents):
    archive, item, context, credentials = patents
    xml = envelope([patent(), lithium_patent(number="2234567", day="20261201")])
    result = collect_patent_signal(item, query_plan(), archive, credentials, context,
        provider_factory=lambda: provider(lambda request: httpx.Response(200, text=xml)))
    assert not result.documents and result.coverage.rejected_records == 2


def test_patent_cancellation_and_time_budget_are_distinct(patents):
    archive, item, context, credentials = patents
    result = collect_patent_signal(item, query_plan(), archive, credentials, context, timeout_seconds=0.000001,
        provider_factory=lambda: provider(lambda request: pytest.fail("Deadline should preclude HTTP")))
    assert result.coverage.limit_reached and "patent_time_budget" in result.coverage.reasons
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        collect_patent_signal(item, query_plan(), archive, credentials, context)


def atom_entry(identity="2501.12345v1", published="2025-01-15T10:00:00Z", updated="2025-01-15T10:00:00Z", **changes):
    fields = dict(title="Lithium membrane performance", summary="A real-shaped preprint metadata fixture.", doi="10.1234/example") | changes
    return f'''<entry><id>https://arxiv.org/abs/{identity}</id><published>{published}</published>
        <updated>{updated}</updated><title>{fields['title']}</title><summary>{fields['summary']}</summary>
        <author><name>A Researcher</name></author><arxiv:doi>{fields['doi']}</arxiv:doi></entry>'''


def write_atom(tmp_path, entries):
    path = tmp_path / "export.atom"
    path.write_text('<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
                    + "".join(entries) + "</feed>")
    return path


def test_local_arxiv_import_keeps_public_date_doi_and_latest_asof_version(tmp_path):
    path = write_atom(tmp_path, [atom_entry(), atom_entry(identity="2501.12345v2", updated="2025-02-01T10:00:00Z")])
    result = import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event())
    assert result.scanned == 2 and result.rejected == 1 and len(result.documents) == 1
    record = result.documents[0]
    assert record.source_id == "2501.12345" and record.url.endswith("v2")
    assert record.publication_date == date(2025, 1, 15) and record.doi == "10.1234/example"
    assert record.raw_metadata["coordinated_export_sha256"] == result.export_sha256
    assert record.document_type == "preprint"


def test_arxiv_future_versions_and_conflicting_first_dates_not_silently_admitted(tmp_path):
    path = write_atom(tmp_path, [atom_entry(), atom_entry(identity="2501.12345v2", updated="2026-05-01T10:00:00Z")])
    result = import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event())
    assert result.documents[0].url.endswith("v1") and result.rejected == 1
    path = write_atom(tmp_path, [atom_entry(), atom_entry(identity="2501.12345v2", published="2025-01-16T10:00:00Z", updated="2025-02-01T00:00:00Z")])
    result = import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event())
    assert not result.documents and result.rejected == 2


@pytest.mark.parametrize("payload", ["not XML", '<!DOCTYPE x [<!ENTITY x SYSTEM "file:///etc/passwd">]><feed>&x;</feed>',
    '<!DOCTYPE feed [<!ENTITY a "123"><!ENTITY b "&a;&a;&a;">]><feed>&b;</feed>', "<feed/>"])
def test_xml_entities_invalid_namespace_and_invalid_xml_rejected(tmp_path, payload):
    path = tmp_path / "bad.xml"
    path.write_text(payload)
    with pytest.raises(EnrichmentError):
        import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event())


def test_arxiv_malformed_metadata_bounds_and_cancellation(tmp_path):
    path = write_atom(tmp_path, [atom_entry(doi="not-a-doi"), atom_entry(), atom_entry(identity="2501.23456")])
    result = import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event(), max_records=2)
    assert result.truncated and result.scanned == 2 and result.rejected == 1 and len(result.documents) == 1
    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=cancel)
    path.write_bytes(b"x" * 5_000_001)
    with pytest.raises(EnrichmentError, match="5 МБ"):
        import_arxiv_atom(path, as_of=date(2026, 1, 1), cancel=Event())

import json

import pytest

from app.backend import __main__ as cli
from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.errors import BackendError
from app.backend.service import Backend


class LocalProvider:
    def iter_pages(self, request, cancel):
        if request.source == "epo":
            raise BackendError("credentials_required", "Configure EPO credentials")
        yield SourcePage(documents=(DocumentRecord(
            source=request.source, source_id="test", doi="10.1234/test",
            title="Test publication", url="https://doi.org/10.1234/test",
        ),), scanned=1, exhausted=True)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    class TestBackend(Backend):
        def __init__(self, settings):
            super().__init__(settings, LocalProvider)
    monkeypatch.setattr(cli, "Backend", TestBackend)


def test_single_source_keeps_original_json_shape(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "collect", "test"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["request"]["source"] == "crossref"
    assert data["state"] == "succeeded" and "jobs" not in data


@pytest.mark.parametrize("sources,code", [(["crossref", "openalex"], 0), (["epo", "openalex"], 1)])
def test_multi_source_json_reports_each_result(tmp_path, capsys, sources, code):
    assert cli.main(["--data-dir", str(tmp_path), "collect", "test", "--sources", *sources]) == code
    data = json.loads(capsys.readouterr().out)
    assert data["all_succeeded"] == (code == 0)
    assert [job["request"]["source"] for job in data["jobs"]] == sources
    assert data["jobs"][-1]["state"] == "succeeded"


def test_local_documents_and_versions(tmp_path, capsys):
    base = ["--data-dir", str(tmp_path)]
    assert cli.main(base + ["collect", "test", "--sources", "crossref", "openalex"]) == 0
    capsys.readouterr()
    assert cli.main(base + ["documents"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 1 and data["items"][0]["sources"] == ["crossref", "openalex"]
    assert cli.main(base + ["versions", "doi:10.1234/test"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2


def test_sources_does_not_create_storage(tmp_path, capsys):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), "sources"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    assert not path.exists()


def test_invalid_query_does_not_create_storage(tmp_path, capsys):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), "collect", "test", "--limit", "0"]) == 2
    assert not path.exists()


def test_primary_topics_cli_request_is_recorded(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "collect", "AI survey area", "--source", "openalex",
                     "--primary-topic-ids", "T2", "https://openalex.org/T1"]) == 0
    request = json.loads(capsys.readouterr().out)["request"]
    assert request["source"] == "openalex"
    assert request["primary_topic_ids"] == ["https://openalex.org/T1", "https://openalex.org/T2"]


@pytest.mark.parametrize("selection", [[], ["--sources", "openalex", "crossref"]])
def test_primary_topics_cli_rejects_unsupported_sources_before_storage(tmp_path, capsys, selection):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), "collect", "AI survey area", *selection,
                     "--primary-topic-ids", "T1"]) == 2
    assert not path.exists()


def test_primary_topics_history_cli_records_selection_in_all_jobs(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "collect-history", "AI survey area",
                     "--sources", "openalex", "--from-date", "2023-01-01", "--until-date", "2024-12-31",
                     "--period", "year", "--limit-per-period", "10000", "--primary-topic-ids", "T1", "T2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["request"]["primary_topic_ids"] == ["https://openalex.org/T1", "https://openalex.org/T2"]
    assert all(period["job"]["request"]["primary_topic_ids"] == report["request"]["primary_topic_ids"]
               for period in report["periods"])

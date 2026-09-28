"""Вкладки панели владельца: источники и страны, языки, анализ анализов, самообучение."""

from __future__ import annotations

from http import HTTPStatus
from http.client import HTTPConnection
import json
from threading import Thread

import pytest

from app.pilot import languages
from app.pilot.approved_sources.catalog import load_policy
from app.pilot.approved_sources.contracts import SOURCE_IDS
from app.relevance_learning import RelevanceModel, load_feedback, train
from app.web_api import ApiHandler, ApiServer, WebApiError
from tests.owner_fixture import make_service


@pytest.fixture
def service(tmp_path):
    return make_service(tmp_path)


def test_sources_policy_is_saved_and_reaches_new_runs(service, tmp_path):
    overview = service.admin_sources()
    assert overview["everywhere"] and len(overview["sources"]) == len(SOURCE_IDS)
    assert service.start_options() == {}
    answer = service.set_sources({"countries": ["RU", "INT"], "disabled": ["gdelt"]})
    assert "обновлены" in answer["message"]
    policy = load_policy(tmp_path)
    assert policy.countries == ("RU", "INT") and policy.disabled == ("gdelt",)
    options = service.start_options()
    assert options["source_policy"]["countries"] == ["RU", "INT"]
    rows = {row["source_id"]: row for row in service.admin_sources()["sources"]}
    assert rows["hal"]["active"] is False and rows["hal"]["skip_reason"] == "country_filter"
    assert rows["cyberleninka"]["active"] and rows["gdelt"]["enabled"] is False
    with pytest.raises(WebApiError):
        service.set_sources({"countries": ["XX"]})
    with pytest.raises(WebApiError):
        service.set_sources({"disabled": sorted(rows)})


def test_languages_are_added_installed_and_made_default(service, tmp_path, monkeypatch):
    installed = []

    def fake_install(data_dir, language, **_options):
        installed.append(language.code)
        spec = {"schema_version": 1, "model_id": language.model_id, "revision": "a" * 40,
                "source_language": "en", "target_language": language.code, "vocabulary_size": 10,
                "decoder_start_token_id": 1, "eos_token_id": 0, "pad_token_id": 1, "beam_width": 2,
                "max_new_tokens": 10, "max_source_tokens": 512, "normalizer_version": languages.PUBLISHED_NORMALIZER,
                "files": [{"name": name, "remote": remote, "sha256": "b" * 64, "bytes": 3}
                          for name, remote in languages.REQUIRED.items()], "optional_files": []}
        directory = languages.language_model_directory(data_dir, language)
        directory.mkdir(parents=True)
        for name in languages.REQUIRED:
            (directory / name).write_bytes(b"abc")
        languages._write_json(languages.spec_path(data_dir, language), spec)
        return {"state": "installed"}

    monkeypatch.setattr(languages, "install_language", fake_install)
    manager = service._language_manager()
    monkeypatch.setattr(manager, "_start_install", lambda language, start: manager._jobs.update(
        {language.code: {"state": "ready"}}) or fake_install(tmp_path, language))
    overview = service.admin_languages()
    assert [row["code"] for row in overview["languages"]] == ["ru"] and overview["default"] == "ru"
    assert any(row["code"] == "de" for row in overview["catalogue"])
    service.set_languages({"op": "add", "code": "de"})
    service.set_languages({"op": "add", "code": "pl", "name": "Польский", "model_id": "Xenova/opus-mt-en-pl"})
    assert installed == ["de", "pl"]
    states = {row["code"]: row["state"] for row in service.admin_languages()["languages"]}
    assert states["de"] == "ready" and states["pl"] == "ready"
    service.set_languages({"op": "default", "code": "de"})
    assert service._reading_language(None) == "de"
    assert service._reading_language("pl") == "pl"
    assert service._reading_language("xx") == "de"
    service.set_languages({"op": "remove", "code": "de"})
    assert service._reading_language(None) == "ru"
    with pytest.raises(WebApiError):
        service.set_languages({"op": "add", "code": "en", "name": "English", "model_id": "a/b"})
    with pytest.raises(WebApiError):
        service.set_languages({"op": "remove", "code": "ru"})


def test_analyses_tab_filters_sorts_and_summarizes(service):
    answer = service.admin_analyses({})
    assert answer["total"] == 6 and answer["index"]["pending"] == 0
    first = answer["runs"][0]
    assert first["id"] == "run-6" and first["state"] == "failed"
    summary = answer["runs"][1]["summary"]
    assert summary["publications"] > 0 and summary["off_topic"] > 0 and summary["technologies"] >= 2
    assert set(summary["countries"]) >= {"RU", "US"}
    statistics = answer["statistics"]
    assert statistics["states"] == {"failed": 1, "succeeded": 5}
    assert statistics["top_queries"][0]["runs"] == 2
    assert statistics["recurring_technologies"]
    assert len(statistics["activity"]) == 30
    only = service.admin_analyses({"q": "квантовые", "state": "succeeded"})
    assert {run["query"] for run in only["runs"]} == {"квантовые сенсоры"}
    deep = service.admin_analyses({"mode": "deep", "sort": "publications", "order": "asc"})
    assert all(run["mode"] == "deep" for run in deep["runs"])
    assert service.admin_analyses({"country": "FR"})["total"] == 5
    with pytest.raises(WebApiError):
        service.admin_analyses({"sort": "nonsense"})
    with pytest.raises(WebApiError):
        service.admin_analyses({"from": "not-a-date"})
    comparison = service.admin_compare(["run-1", "run-4"])
    assert comparison["common_technologies"] and len(comparison["analyses"]) == 2
    with pytest.raises(WebApiError):
        service.admin_compare(["run-1"])


def test_publications_view_and_feedback_teach_the_model(service, tmp_path):
    view = service.admin_publications({"run_id": "run-1", "decision": "off_topic"})
    assert view["rated"] and view["counts"]["off_topic"] == view["total"] > 0
    noise = view["publications"][0]
    assert noise["decision"] == "off_topic" and noise["mark"] is None
    answer = service.set_feedback({"run_id": "run-1", "publication_id": noise["publication_id"], "label": 1})
    assert answer["label"] == 1 and "по теме" in answer["message"]
    assert load_feedback(tmp_path)["run-1"][noise["publication_id"]]["label"] == 1
    marked = service.admin_publications({"run_id": "run-1", "decision": "marked"})
    assert [item["mark"] for item in marked["publications"]] == [1]
    service.set_feedback({"run_id": "run-1", "publication_id": noise["publication_id"], "label": None})
    assert "run-1" not in load_feedback(tmp_path)
    with pytest.raises(WebApiError):
        service.set_feedback({"run_id": "run-1", "publication_id": "0" * 64, "label": 1})


def test_training_on_own_analyses_beats_nothing_and_reports_quality(service, tmp_path):
    model = train(tmp_path)
    assert model.samples >= 200 and model.runs == 6
    holdout = model.metrics["holdout"]
    assert holdout["auc"] >= 0.9 and holdout["samples"] > 0
    assert model.weights and model.version
    loaded = RelevanceModel.load(tmp_path)
    assert loaded.version == model.version
    learning = service.admin_learning()
    assert learning["samples"] == model.samples and learning["history"]
    assert learning["sources"]["google_news"]["items"] > 0
    assert set(learning["settings"]) == {"enabled", "auto_retrain", "adapt_sources"}
    service.learning_action("learning", {"enabled": False})
    assert RelevanceModel.load(tmp_path).ready is False
    service.learning_action("learning", {"enabled": True})
    service.learning_action("reset_learning", {})
    assert service.admin_learning()["samples"] == 0


def test_owner_views_need_the_admin_token_and_valid_parameters(service, monkeypatch):
    monkeypatch.setenv("TREND_API_TOKEN", "t" * 40)
    monkeypatch.setenv("TREND_API_ADMIN_TOKEN", "a" * 40)
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port

    def request(method, path, body=None, token="a" * 40):
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Host": f"127.0.0.1:{port}"}
        if token:
            headers["X-Trend-Admin-Token"] = token
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body)
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    try:
        assert request("GET", "/admin/sources", token=None)[0] == HTTPStatus.FORBIDDEN
        assert request("GET", "/admin/sources", token="t" * 40)[0] == HTTPStatus.FORBIDDEN
        status, answer = request("GET", "/admin/analyses?q=%D0%BA%D0%B2%D0%B0%D0%BD%D1%82&limit=5")
        assert status == HTTPStatus.OK and answer["total"] == 2
        assert request("GET", "/admin/analyses?unknown=1")[0] == HTTPStatus.UNPROCESSABLE_ENTITY
        assert request("GET", "/admin/compare?ids=run-1,run-2")[0] == HTTPStatus.OK
        assert request("GET", "/admin/publications?run_id=run-2&decision=relevant")[0] == HTTPStatus.OK
        assert request("GET", "/admin/learning")[0] == HTTPStatus.OK
        status, answer = request("POST", "/admin/action", {"action": "sources", "values": {"countries": ["RU"]}})
        assert status == HTTPStatus.OK and answer["sources"]["policy"]["countries"] == ["RU"]
        assert request("POST", "/admin/action", {"action": "sources"})[0] == HTTPStatus.UNPROCESSABLE_ENTITY
        status, _ = request("POST", "/admin/action", {"action": "retrain", "values": {}})
        assert status == HTTPStatus.OK
    finally:
        server.shutdown()
        server.server_close()

"""Автоматическая раскладка моделей при запуске: только локально, без тихой загрузки."""

import socket

import pytest

from app.pilot import service as service_module
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure


@pytest.fixture
def placements(monkeypatch):
    """Record every staged placement the service attempts, without copying weights."""
    calls = []

    def install_from_staging(key, directory, spec, verify):
        calls.append((key, str(directory)))
        return False

    monkeypatch.setattr("scripts.model_staging.install_from_staging", install_from_staging)
    return calls


@pytest.fixture
def no_network(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("Запуск не должен обращаться к сети")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture
def service(tmp_path, monkeypatch):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _name: None)
    instance = PilotService(tmp_path / "data", credentials)
    try:
        yield instance
    finally:
        instance.close()


def test_preparation_places_both_local_models_into_this_profile(service, placements, no_network):
    report = service.prepare_local_models()
    assert report["origin"] == "development"
    assert [key for key, _ in placements] == ["multilingual-e5-small", "opus-mt-ru-en"]
    assert placements[0][1] == str(service.model_dir), "научная модель кладётся в профиль службы"
    assert placements[1][1].startswith(str(service.data_dir)), "переводчик тоже в профиль службы"


def test_reading_the_status_never_copies_anything(service, placements, no_network):
    """status() answers a question; it must not change the profile as a side effect."""
    assert service.status()["model_installed"] is False
    assert placements == []


def test_an_existing_directory_is_left_alone(service, placements):
    service.model_dir.mkdir(parents=True)
    service.prepare_local_models()
    assert [key for key, _ in placements] == ["opus-mt-ru-en"], "занятый каталог не трогаем"


def test_a_failing_placement_is_not_an_error_but_a_missing_model(service, monkeypatch, no_network):
    def broken(*_args, **_kwargs):
        raise OSError("диск недоступен")

    monkeypatch.setattr("scripts.model_staging.install_from_staging", broken)
    assert service.prepare_local_models()["placed"] == {}
    assert service.status()["model_installed"] is False


def test_a_bundled_application_never_places_anything_itself(service, placements, monkeypatch):
    """A frozen build already carries its models; copying into a profile would be wrong."""
    monkeypatch.setattr(service, "model_location",
                        service.model_location.__class__(service.model_dir, "bundled",
                                                         "multilingual-e5-small", "a" * 40))
    report = service.prepare_local_models()
    assert report["origin"] == "bundled" and report["placed"] == {}
    assert placements == []


def test_translate_query_reports_a_missing_model_without_copying(service, placements, no_network):
    with pytest.raises(TaskFailure):
        service.translate_query("Селективные мембраны")
    assert placements == [], "перевод — не момент для раскладки моделей"


def test_translate_query_still_refuses_an_empty_direction(service, placements):
    with pytest.raises(TaskFailure, match="Введите направление"):
        service.translate_query("   ")
    assert placements == []


def test_preparation_is_a_published_service_operation():
    assert hasattr(service_module.PilotService, "prepare_local_models")

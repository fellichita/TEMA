"""Локальные копии моделей: вторая установка не выходит в сеть и не верит имени каталога."""

import json

import pytest

from scripts import model_staging
from scripts.model_staging import (
    install_from_staging, stage_from, staging_directory, verified_staged_copy,
)

SPEC = {"files": [{"name": "model.bin", "bytes": 4, "sha256": "x"},
                  {"name": "tokenizer.json", "bytes": 2, "sha256": "y"}]}


def write(directory, *, good=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(b"good" if good else b"bad!")
    (directory / "tokenizer.json").write_bytes(b"ok")
    return directory


def verifier(calls=None):
    def verify(path, spec):
        if calls is not None:
            calls.append(path)
        if (path / "model.bin").read_bytes() != b"good":
            raise ValueError("Содержимое не совпадает с закреплённым")
        return path
    return verify


@pytest.fixture
def project(tmp_path, monkeypatch):
    lock = {"schema_version": 1,
            "models": [{"key": "demo", "source": "storage/demo-models/demo"}],
            "notices": []}
    (tmp_path / "resources/models").mkdir(parents=True)
    (tmp_path / "resources/models/registry.json").write_text(
        json.dumps(lock, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(model_staging, "PROJECT", tmp_path)
    monkeypatch.setattr(model_staging, "LOCK", tmp_path / "resources/models/registry.json")
    return tmp_path


def test_staging_path_comes_from_the_lock_and_stays_inside_the_project(project):
    assert staging_directory("demo") == project / "storage/demo-models/demo"
    with pytest.raises(ValueError, match="Unknown model key"):
        staging_directory("absent")


@pytest.mark.parametrize("source", ["/etc/passwd", "../outside", "storage/../../escape"])
def test_a_lock_pointing_outside_the_project_is_refused(project, source):
    (project / "resources/models/registry.json").write_text(
        json.dumps({"schema_version": 1, "models": [{"key": "demo", "source": source}], "notices": []}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid staged model source"):
        staging_directory("demo")


def test_absent_or_damaged_stage_is_not_used_and_is_not_an_error(project):
    assert verified_staged_copy("demo", SPEC, verifier()) is None
    write(project / "storage/demo-models/demo", good=False)
    assert verified_staged_copy("demo", SPEC, verifier()) is None
    assert install_from_staging("demo", project / "profile/demo", SPEC, verifier()) is False
    assert not (project / "profile/demo").exists(), "повреждённый стенд не публикуется"


def test_install_copies_locally_and_verifies_before_and_after(project):
    write(project / "storage/demo-models/demo")
    target = project / "profile/demo"
    calls = []
    assert install_from_staging("demo", target, SPEC, verifier(calls)) is True
    assert (target / "model.bin").read_bytes() == b"good"
    assert len(calls) == 2, "проверка и источника, и опубликованной копии"


def test_a_non_empty_destination_is_left_to_the_caller(project):
    write(project / "storage/demo-models/demo")
    target = write(project / "profile/demo", good=False)
    assert install_from_staging("demo", target, SPEC, verifier()) is False
    assert (target / "model.bin").read_bytes() == b"bad!", "чужой каталог не перезаписан"


def test_the_profile_copy_fills_an_empty_project_stage(project):
    source = write(project / "elsewhere/demo")
    assert stage_from("demo", source, SPEC, verifier()) is True
    assert verified_staged_copy("demo", SPEC, verifier()) is not None
    assert stage_from("demo", source, SPEC, verifier()) is False, "повторное сохранение не нужно"


def test_an_unverified_directory_never_becomes_the_project_stage(project):
    source = write(project / "elsewhere/demo", good=False)
    with pytest.raises(ValueError, match="закреплённым"):
        stage_from("demo", source, SPEC, verifier())
    assert not (project / "storage/demo-models/demo").exists()

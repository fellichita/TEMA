"""Source identity includes checkout rules, runtime resources and launch inputs."""

import pytest

from tools import run_checks


def test_checkout_rules_change_source_identity(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    monkeypatch.setattr(run_checks, "REPO", tmp_path)
    without_attributes = run_checks.source_fingerprint()
    attributes = tmp_path / ".gitattributes"
    attributes.write_text("* text=auto eol=lf\n")
    with_lf = run_checks.source_fingerprint()
    attributes.write_text("* text=auto eol=crlf\n")
    with_crlf = run_checks.source_fingerprint()
    assert len({without_attributes, with_lf, with_crlf}) == 3


def test_source_identity_does_not_depend_on_platform_path_sort_order(tmp_path, monkeypatch):
    class CaseSensitiveOrderedPath(type(tmp_path)):
        def __lt__(self, other):
            return self.as_posix() < other.as_posix()

    class CaseFoldOrderedPath(type(tmp_path)):
        """Use Windows ordering while reading the same local fixture bytes."""

        def __lt__(self, other):
            return self.as_posix().casefold() < other.as_posix().casefold()

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    app = tmp_path / "app"
    app.mkdir()
    (app / "Zeta.py").write_text("UPPER = 1\n")
    (app / "alpha.py").write_text("lower = 2\n")
    monkeypatch.setattr(run_checks, "REPO", CaseSensitiveOrderedPath(tmp_path))
    case_sensitive = run_checks.source_fingerprint()
    monkeypatch.setattr(run_checks, "REPO", CaseFoldOrderedPath(tmp_path))
    assert run_checks.source_fingerprint() == case_sensitive


@pytest.mark.parametrize('name', [
    'requirements/base.lock', 'requirements/web.txt', 'resources/models/registry.json',
    'launchers/macos/desktop.command', 'launchers/windows/web-demo.bat',
])
def test_launch_or_runtime_input_changes_source_identity(tmp_path, monkeypatch, name):
    (tmp_path / 'pyproject.toml').write_text('[project]\n')
    source = tmp_path / name
    source.parent.mkdir(parents=True)
    source.write_text('original runtime input')
    monkeypatch.setattr(run_checks, 'REPO', tmp_path)
    original = run_checks.source_fingerprint()
    source.write_text('changed runtime input')
    assert run_checks.source_fingerprint() != original

"""Source launch must be local, repeatable, and safe for an existing checkout."""

from pathlib import Path
from contextlib import nullcontext
import subprocess
import sys
import shutil

import pytest

from scripts import mac_bootstrap as bootstrap
from tests.platform_support import require_symlinks


def test_subprocess_preserves_special_paths_and_arguments(tmp_path):
    directory = tmp_path / "Другой Mac $literal `literal` ' пробел"
    directory.mkdir()
    argument = "$(touch should-not-exist); ' `hello`"
    result = bootstrap.run([sys.executable, "-c", "import sys; print(sys.argv[1])", argument],
                           directory, capture=True)
    assert result.returncode == 0
    assert result.stdout.strip() == argument
    assert list(directory.iterdir()) == []


def test_installer_does_not_inherit_other_python_or_package_indexes(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/other/project")
    monkeypatch.setenv("PYTHONHOME", "/other/python")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://private.example/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://private.example/simple")
    env = bootstrap.child_environment()
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert "PIP_INDEX_URL" not in env and "PIP_EXTRA_INDEX_URL" not in env
    assert env["PIP_NO_INPUT"] == "1"
    assert env["ORT_DISABLE_TELEMETRY"] == "1"


@pytest.mark.parametrize("machine,version,expected", [
    ("x86_64", "15.2", "Intel"), ("arm64", "13.7", "macOS 14"),
])
def test_rejects_incompatible_mac_before_download(monkeypatch, machine, version, expected):
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: machine)
    monkeypatch.setattr(bootstrap.platform, "mac_ver", lambda: (version, (), ""))
    with pytest.raises(bootstrap.SetupError, match=expected):
        bootstrap.validate_platform()


def test_rejects_free_threaded_python(monkeypatch):
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")
    monkeypatch.setattr(bootstrap.sysconfig, "get_config_var", lambda key: 1)
    with pytest.raises(bootstrap.SetupError, match="free-threaded"):
        bootstrap.validate_platform()


def test_check_does_not_install_or_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "validate_platform", lambda: None)
    monkeypatch.setattr(bootstrap, "installation_lock", lambda _: nullcontext())
    monkeypatch.setattr(bootstrap, "install", lambda _: pytest.fail("Check attempted installation"))
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda *a, **k: pytest.fail("Check launched GUI"))
    assert bootstrap.main(["--check"]) == 1
    assert not (tmp_path / ".venv").exists()


def test_ready_environment_starts_without_install_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "validate_platform", lambda: None)
    monkeypatch.setattr(bootstrap, "installation_lock", lambda _: nullcontext())
    monkeypatch.setattr(bootstrap, "runtime_status", lambda _: (True, "Runtime ready"))
    monkeypatch.setattr(bootstrap, "install", lambda _: pytest.fail("Ready environment was reinstalled"))
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda *a, **k: calls.append((a, k)) or 0)
    assert bootstrap.main(["--", "--data-dir", "storage/Новая библиотека"]) == 0
    command = calls[0][0][0]
    assert command[-2:] == ["--data-dir", "storage/Новая библиотека"]
    assert calls[0][1]["cwd"] == tmp_path


def test_installation_lock_blocks_second_installer_and_recovers(tmp_path):
    if sys.platform == "win32":
        with pytest.raises(bootstrap.SetupError, match="Windows"):
            with bootstrap.installation_lock(tmp_path):
                pytest.fail("Windows entered a POSIX installer lock")
        return
    with bootstrap.installation_lock(tmp_path):
        with pytest.raises(bootstrap.SetupError, match="уже идёт"):
            with bootstrap.installation_lock(tmp_path):
                pytest.fail("Second installer entered lock")
    with bootstrap.installation_lock(tmp_path):
        pass


def test_existing_broken_environment_is_preserved(tmp_path, monkeypatch):
    directory = tmp_path / ".venv"
    directory.mkdir()
    sentinel = directory / "keep-me"
    sentinel.write_text("existing files", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""))
    with pytest.raises(bootstrap.SetupError, match="Переименуйте"):
        bootstrap.install(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "existing files"


def test_failed_install_stops_before_next_stage(tmp_path, monkeypatch):
    directory = tmp_path / ".venv" / "bin"
    directory.mkdir(parents=True)
    (directory / "python").touch()
    calls = []

    def execute(command, root, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if "install" in command else 0, "", "")

    monkeypatch.setattr(bootstrap, "run", execute)
    with pytest.raises(bootstrap.SetupError, match="Установка остановлена"):
        bootstrap.install(tmp_path)
    assert len(calls) == 3
    assert not any("scripts.build_sqlite_runtime" in command for command in calls)


def test_symlink_environment_is_never_modified(tmp_path, monkeypatch):
    require_symlinks()
    other = tmp_path / "Original project environment"
    other.mkdir()
    (tmp_path / ".venv").symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(bootstrap, "run", lambda *a, **k: pytest.fail("Touched shared environment"))
    with pytest.raises(bootstrap.SetupError, match="ссылкой"):
        bootstrap.install(tmp_path)
    assert list(other.iterdir()) == []


def test_plain_system_python_in_venv_folder_never_receives_packages(tmp_path, monkeypatch):
    directory = tmp_path / ".venv" / "bin"
    directory.mkdir(parents=True)
    (directory / "python").touch()
    calls = []

    def execute(command, root, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if "-c" in command else 0, "", "")

    monkeypatch.setattr(bootstrap, "run", execute)
    with pytest.raises(bootstrap.SetupError, match="отдельным окружением"):
        bootstrap.install(tmp_path)
    assert len(calls) == 2
    assert "sys.prefix != sys.base_prefix" in calls[-1][-2]
    assert not any("install" in command for command in calls)


def test_finder_launcher_is_valid_shell():
    launcher = Path(__file__).resolve().parents[1] / "launchers/macos/desktop.command"
    shell = Path("/bin/bash")
    if sys.platform == "win32":
        # git.exe usually resolves inside mingw64in, where bash does not live;
        # the real shells sit in the installation's own bin and usr/bin.
        found = shutil.which("bash")
        candidates = [Path(found)] if found else []
        git = shutil.which("git")
        if git is not None:
            root = Path(git).resolve().parent
            candidates += [root / "bash.exe", *(parent / folder / "bash.exe"
                                                for parent in (root.parent, root.parent.parent)
                                                for folder in ("bin", "usr/bin"))]
        shell = next((path for path in candidates if path.is_file()), None)
        assert shell is not None, "Git for Windows with Bash is required to syntax-check the Finder launcher"
    result = subprocess.run([str(shell), "-n", launcher.name], cwd=launcher.parent,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,target", [
    ("desktop.command", "scripts/mac_bootstrap.py"),
    ("web-local.command", "scripts.local_web_bootstrap"),
    ("web-demo.command", "scripts.run_web_demo"),
])
def test_nested_launchers_use_project_root_and_preserve_arguments(tmp_path, name, target):
    if sys.platform == "win32":
        return  # These launchers execute only on macOS; Windows checks syntax above.
    root = tmp_path / "Проект с пробелом ' $literal"
    launchers = root / "launchers/macos"
    launchers.mkdir(parents=True)
    launcher = launchers / name
    shutil.copy2(Path(__file__).resolve().parents[1] / "launchers/macos" / name, launcher)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nif [ "$3" = "-c" ]; then exit 0; fi\n'
                      'printf "%s\\n" "$PWD" "$@" > launcher-output.txt\n', encoding="utf-8")
    python.chmod(0o755)
    argument = "$(touch should-not-exist); ' `literal`"
    result = subprocess.run(["/bin/bash", str(launcher), argument], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = (root / "launcher-output.txt").read_text(encoding="utf-8").splitlines()
    assert output[0] == str(root)
    assert output[-1] == argument
    assert any(value.endswith(target) for value in output)
    assert not (root / "should-not-exist").exists()

"""A complete public scan and pinned native proof are both required for CI."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts import audit_runtime_dependencies as audit
from tests.platform_support import require_symlinks


INVENTORY = {"pysqlite3": audit.builder.DRIVER_VERSION, "public-example": "1.2.3"}
PUBLIC_REPORT = {"dependencies": [{"name": "public-example", "version": "1.2.3", "vulns": []}]}


@pytest.mark.parametrize("private", [None, "0.6.0", "0.6.0+sqlite3.53.1", "0.6.0+unknown"])
def test_missing_or_unknown_private_sqlite_is_never_skipped(private):
    inventory = {"public-example": "1.2.3"}
    if private is not None:
        inventory["pysqlite3"] = private
    with pytest.raises(audit.AuditFailure, match="private_sqlite_version_mismatch"):
        audit.public_inventory(inventory)


def test_another_local_distribution_requires_its_own_verification():
    with pytest.raises(audit.AuditFailure, match="unverified_private_distribution"):
        audit.public_inventory({**INVENTORY, "other-private": "1.0+local"})


@pytest.mark.parametrize("dependencies", [
    [],
    [{"name": "public-example", "version": "1.2.3", "skip_reason": "not in PyPI"}],
    [{"name": "public-example", "version": "1.2.3", "vulns": [], "skip_reason": ""}],
    [{"name": "public-example", "version": "1.2.3", "vulns": [{"id": "CVE-test"}]}],
    [{"name": "public-example", "version": "1.2.2", "vulns": []}],
    [{"name": "different-package", "version": "1.2.3", "vulns": []}],
    [{"name": "public-example", "version": "1.2.3"}],
    PUBLIC_REPORT["dependencies"] * 2,
])
def test_success_exit_cannot_hide_skips_vulnerabilities_missing_packages_or_changed_versions(dependencies):
    with pytest.raises(audit.AuditFailure):
        audit.validate_public_report({"dependencies": dependencies}, {"public-example": "1.2.3"})


def test_exact_public_result_can_pass():
    audit.validate_public_report(PUBLIC_REPORT, {"public-example": "1.2.3"})


def test_auditor_uses_synthetic_home_without_inheriting_user_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", "C:\\private\\real-user")
    monkeypatch.setenv("APPDATA", "C:\\private\\real-user\\AppData\\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", "C:\\private\\real-user\\AppData\\Local")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-sentinel")
    home = tmp_path / "auditor-home"

    environment = audit.auditor_environment(home)

    assert environment["HOME"] == str(home)
    assert environment["USERPROFILE"] == str(home)
    assert environment["APPDATA"] == str(home / "AppData" / "Roaming")
    assert environment["LOCALAPPDATA"] == str(home / "AppData" / "Local")
    assert "DEEPSEEK_API_KEY" not in environment
    assert all("real-user" not in environment[name]
               for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"))


def setup_audit(tmp_path, monkeypatch, *, returncode=0, report=PUBLIC_REPORT):
    monkeypatch.setattr(audit, "installed_inventory", lambda: INVENTORY)
    monkeypatch.setattr(audit, "verify_private_sqlite", lambda root: {
        "version": audit.builder.DRIVER_VERSION, "sqlite_version": audit.builder.SQLITE_VERSION,
    })
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-sentinel")
    monkeypatch.setenv("PYTHONPATH", "/private/developer/path")
    calls = []

    def scan(command, *, env, capture_output, text, timeout, check):
        calls.append(command)
        assert "--strict" in command and "--disable-pip" in command and "--no-deps" in command
        assert command[1:4] == ["-I", "-B", "-m"]
        assert "DEEPSEEK_API_KEY" not in env and "PYTHONPATH" not in env
        assert capture_output and text and timeout == 300 and check is False
        requirement_file = Path(command[command.index("-r") + 1])
        assert requirement_file.read_text(encoding="utf-8") == "public-example==1.2.3\n"
        assert Path(env["USERPROFILE"]) == requirement_file.parent / "auditor-home"
        assert Path(env["USERPROFILE"]).is_dir()
        result = Path(command[command.index("--output") + 1])
        result.write_text(json.dumps(report))
        return SimpleNamespace(returncode=returncode, stdout="", stderr="secret-sentinel")

    monkeypatch.setattr(audit.subprocess, "run", scan)
    return calls


def test_full_audit_excludes_only_verified_private_wheel_and_scans_exact_public_versions(tmp_path, monkeypatch):
    calls = setup_audit(tmp_path, monkeypatch)
    result = audit.run_audit(tmp_path / "auditor-python", tmp_path / "report", tmp_path / "build")
    assert len(calls) == 1
    assert result["status"] == "passed" and result["public_count"] == 1
    assert result["public_skips"] == 0 and result["verified_private_count"] == 1
    assert "pysqlite3" in json.loads((tmp_path / "report/installed-inventory.json").read_text(encoding="utf-8"))
    assert (tmp_path / "report/private-sqlite-proof.json").is_file()


def test_auditor_venv_symlink_keeps_its_environment_identity(tmp_path, monkeypatch):
    require_symlinks()
    calls = setup_audit(tmp_path, monkeypatch)
    interpreter = tmp_path / "base-python"
    interpreter.write_text("test executable marker")
    launcher = tmp_path / "auditor-venv-python"
    try:
        launcher.symlink_to(interpreter)
    except OSError:
        # Windows may prohibit symlink creation for unprivileged users; Path's
        # resolved target can be substituted without requiring that permission.
        original_resolve = Path.resolve
        monkeypatch.setattr(Path, "resolve", lambda self, *args, **kwargs:
                            interpreter if self == launcher else original_resolve(self, *args, **kwargs))
    audit.run_audit(launcher, tmp_path / "report", tmp_path / "build")
    assert calls[0][0] == str(launcher.absolute())


@pytest.mark.parametrize(("returncode", "report"), [(2, PUBLIC_REPORT), (0, {"dependencies": []})])
def test_incomplete_scan_overwrites_previous_pass_and_never_logs_sensitive_stderr(tmp_path, monkeypatch, returncode, report):
    setup_audit(tmp_path, monkeypatch, returncode=returncode, report=report)
    directory = tmp_path / "report"
    directory.mkdir()
    (directory / "audit-summary.json").write_text('{"status":"passed"}')
    with pytest.raises(audit.AuditFailure):
        audit.run_audit(tmp_path / "auditor-python", directory, tmp_path / "build")
    summary = (directory / "audit-summary.json").read_text(encoding="utf-8")
    assert json.loads(summary)["status"] == "failed"
    assert "secret-sentinel" not in summary


def test_failed_private_proof_stops_before_network_scan(tmp_path, monkeypatch):
    calls = setup_audit(tmp_path, monkeypatch)

    def fail(_):
        raise audit.AuditFailure("installed_private_files_do_not_match_built_wheel")

    monkeypatch.setattr(audit, "verify_private_sqlite", fail)
    with pytest.raises(audit.AuditFailure):
        audit.run_audit(tmp_path / "auditor-python", tmp_path / "report", tmp_path / "build")
    assert calls == []


def test_checkout_cannot_shadow_isolated_auditor_with_local_module(tmp_path, monkeypatch):
    marker = tmp_path / "shadow-module-executed"
    (tmp_path / "pip_audit.py").write_text(
        "from pathlib import Path\nPath('shadow-module-executed').write_text('not isolated')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, "installed_inventory", lambda: INVENTORY)
    monkeypatch.setattr(audit, "verify_private_sqlite", lambda _: {
        "version": audit.builder.DRIVER_VERSION, "sqlite_version": audit.builder.SQLITE_VERSION,
    })
    real_run = audit.subprocess.run

    def local_probe(command, **kwargs):
        # Execute the actual interpreter flags. --help makes the check offline
        # even if this test interpreter happens to include a real pip-audit.
        return real_run([*command, "--help"], **kwargs)

    monkeypatch.setattr(audit.subprocess, "run", local_probe)
    with pytest.raises(audit.AuditFailure):
        audit.run_audit(Path(sys.executable), tmp_path / "report", tmp_path / "build")
    assert not marker.exists(), "A checkout module replaced the isolated auditor"


def fixture_sources(tmp_path, monkeypatch, source_id):
    directory = tmp_path / "runtime-sources"
    directory.mkdir()
    driver = directory / "driver-source.tar.gz"
    driver.write_bytes(b"synthetic driver fixture; not compiled")
    archive = directory / "sqlite-source.zip"
    code = f'#define SQLITE_SOURCE_ID "{source_id}"\n'.encode("ascii")
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("sqlite-amalgamation/sqlite3.c", code)
    monkeypatch.setattr(audit.builder, "SOURCES", tuple(
        (path.name, "https://invalid.example/never-requested", audit.digest(path)) for path in (driver, archive)))
    monkeypatch.setattr(audit.builder, "SQLITE_ARCHIVE_SHA3_256", audit.digest(archive, "sha3_256"))
    monkeypatch.setattr(audit.builder, "SQLITE_C_SHA3_256", hashlib.sha3_256(code).hexdigest())
    return driver


def test_private_source_checksum_is_checked_before_wheel_lookup(tmp_path, monkeypatch):
    driver = fixture_sources(tmp_path, monkeypatch, "fixture source")
    driver.write_bytes(b"changed after pinning")
    with pytest.raises(audit.AuditFailure, match="private_source_sha256_mismatch"):
        audit.verify_private_sqlite(tmp_path)


def test_source_pin_does_not_substitute_for_checking_loaded_sqlite_identity(tmp_path, monkeypatch):
    fixture_sources(tmp_path, monkeypatch, "different source from loaded SQLite")
    with pytest.raises(audit.AuditFailure, match="loaded_sqlite_differs_from_pinned_source"):
        audit.verify_private_sqlite(tmp_path)


def test_source_and_runtime_match_still_require_matching_built_wheel(tmp_path, monkeypatch):
    private = audit.importlib.import_module("pysqlite3")
    with private.connect(":memory:") as connection:
        source_id = connection.execute("SELECT sqlite_source_id()").fetchone()[0]
    fixture_sources(tmp_path, monkeypatch, source_id)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    with zipfile.ZipFile(wheels / f"pysqlite3-{audit.builder.DRIVER_VERSION}-fake.whl", "w") as wheel:
        wheel.writestr("pysqlite3/incorrect.py", b"changed wrapper")
    with pytest.raises(audit.AuditFailure, match="installed_private_files_do_not_match_built_wheel"):
        audit.verify_private_sqlite(tmp_path)

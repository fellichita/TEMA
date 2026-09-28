"""Audit the active runtime completely, including its separately verified SQLite.

Run with the application interpreter and pass an isolated pip-audit interpreter.
The private wheel is verified against pinned sources and loaded code; it is never
renamed to a public PyPI version or silently skipped by the vulnerability scanner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import zipfile

from packaging.utils import canonicalize_name
from packaging.version import Version

from scripts import build_sqlite_runtime as builder


class AuditFailure(ValueError):
    """A fixed diagnostic code, without provider output or environment values."""


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def installed_inventory() -> dict[str, str]:
    names = {canonicalize_name(dist.metadata["Name"]) for dist in importlib.metadata.distributions()}
    # Resolve through the active interpreter's search order. A read-through .pth
    # may expose older metadata after the selected private distribution.
    return {name: importlib.metadata.version(name) for name in sorted(names)}


def public_inventory(inventory: dict[str, str]) -> dict[str, str]:
    if inventory.get("pysqlite3") != builder.DRIVER_VERSION:
        raise AuditFailure("private_sqlite_version_mismatch")
    for name, version in inventory.items():
        if not re.fullmatch(r"[a-z0-9]+(?:[-][a-z0-9]+)*", name):
            raise AuditFailure("invalid_distribution_name")
        parsed = Version(version)
        if name != "pysqlite3" and parsed.local is not None:
            raise AuditFailure("unverified_private_distribution")
    return {name: version for name, version in inventory.items() if name != "pysqlite3"}


def verify_private_sqlite(build_root: Path) -> dict[str, Any]:
    from app.sqlite_runtime import MIN_SQLITE_VERSION, sqlite3 as selected

    private = importlib.import_module("pysqlite3")
    native = importlib.import_module("pysqlite3._sqlite3")
    distribution = importlib.metadata.distribution("pysqlite3")
    if distribution.version != builder.DRIVER_VERSION or private.sqlite_version != builder.SQLITE_VERSION:
        raise AuditFailure("private_sqlite_version_mismatch")
    archives = []
    for name, url, expected in builder.SOURCES:
        path = build_root / "runtime-sources" / name
        actual = digest(path)
        if actual != expected:
            raise AuditFailure("private_source_sha256_mismatch")
        archives.append({"name": name, "url": url, "sha256": actual})
    sqlite_archive = build_root / "runtime-sources" / builder.SOURCES[1][0]
    if digest(sqlite_archive, "sha3_256") != builder.SQLITE_ARCHIVE_SHA3_256:
        raise AuditFailure("sqlite_archive_official_hash_mismatch")
    with zipfile.ZipFile(sqlite_archive) as archive:
        sources = [name for name in archive.namelist() if name.endswith("/sqlite3.c")]
        if len(sources) != 1:
            raise AuditFailure("sqlite_source_missing")
        code = archive.read(sources[0])
    if hashlib.sha3_256(code).hexdigest() != builder.SQLITE_C_SHA3_256:
        raise AuditFailure("sqlite_c_official_hash_mismatch")
    source_id = re.search(rb'^#define SQLITE_SOURCE_ID +"([^"]+)"', code, re.MULTILINE)
    if source_id is None:
        raise AuditFailure("sqlite_source_id_missing")
    with private.connect(":memory:") as connection:
        actual_id = connection.execute("SELECT sqlite_source_id()").fetchone()[0]
        connection.execute("CREATE VIRTUAL TABLE audit_fts USING fts5(text)")
        connection.execute("INSERT INTO audit_fts VALUES ('verified runtime')")
        count = connection.execute("SELECT count(*) FROM audit_fts WHERE audit_fts MATCH 'verified'").fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if actual_id != source_id.group(1).decode("ascii") or count != 1 or integrity != "ok":
        raise AuditFailure("loaded_sqlite_differs_from_pinned_source")
    if selected.sqlite_version_info < MIN_SQLITE_VERSION:
        raise AuditFailure("selected_sqlite_below_security_floor")
    files = {
        str(item).replace("\\", "/"): Path(str(distribution.locate_file(item)))
        for item in distribution.files or ()
        if str(item).replace("\\", "/").startswith("pysqlite3/")
        and not str(item).endswith(".pyc")
    }
    if not files or not isinstance(native.__file__, str) or Path(native.__file__).resolve() not in {
            path.resolve() for path in files.values()}:
        raise AuditFailure("loaded_private_module_not_owned_by_distribution")
    installed = {name: digest(path) for name, path in files.items()}
    matched_wheel = None
    for wheel in sorted((build_root / "wheels").glob(f"pysqlite3-{builder.DRIVER_VERSION}-*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            members = {name for name in archive.namelist() if name.startswith("pysqlite3/") and not name.endswith("/")}
            if members == set(installed) and all(
                    hashlib.sha256(archive.read(name)).hexdigest() == installed[name] for name in members):
                matched_wheel = wheel
                break
    if matched_wheel is None:
        raise AuditFailure("installed_private_files_do_not_match_built_wheel")
    return {"distribution": "pysqlite3", "version": distribution.version,
            "sqlite_version": private.sqlite_version, "sqlite_source_id": actual_id,
            "selected_sqlite_version": selected.sqlite_version,
            "minimum_secure_version": ".".join(map(str, MIN_SQLITE_VERSION)),
            "source_archives": archives, "sqlite_archive_sha3_256": builder.SQLITE_ARCHIVE_SHA3_256,
            "sqlite_c_sha3_256": builder.SQLITE_C_SHA3_256,
            "wheel": matched_wheel.name, "wheel_sha256": digest(matched_wheel),
            "installed_files_sha256": installed, "fts5": True, "integrity": integrity,
            "verification": "pinned_source_and_loaded_runtime_and_wheel"}


def validate_public_report(report: dict[str, Any], expected: dict[str, str]) -> None:
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list):
        raise AuditFailure("invalid_public_audit_report")
    actual: dict[str, str] = {}
    for dependency in dependencies:
        if not isinstance(dependency, dict) or "skip_reason" in dependency:
            raise AuditFailure("public_distribution_skipped")
        name = canonicalize_name(dependency.get("name", ""))
        if name in actual or dependency.get("vulns") != []:
            raise AuditFailure("public_vulnerability_or_invalid_result")
        actual[name] = dependency.get("version", "")
    if actual != expected:
        raise AuditFailure("public_audit_inventory_mismatch")


def auditor_environment(home: Path) -> dict[str, str]:
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    # pip-audit resolves Path.home() while importing, before --cache-dir is
    # parsed. Windows needs USERPROFILE for this even in an isolated process.
    # Give the auditor a fresh profile instead of the real user's settings.
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_NO_CACHE_DIR="1",
                       HOME=str(home), USERPROFILE=str(home),
                       APPDATA=str(home / "AppData" / "Roaming"),
                       LOCALAPPDATA=str(home / "AppData" / "Local"))
    return environment


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_audit(auditor_python: Path, output_dir: Path, build_root: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audit-summary.json"
    write_json(summary_path, {"status": "running"})
    try:
        inventory = installed_inventory()
        write_json(output_dir / "installed-inventory.json", inventory)
        public = public_inventory(inventory)
        private = verify_private_sqlite(build_root)
        write_json(output_dir / "private-sqlite-proof.json", private)
        requirements = output_dir / "installed-public-exact.txt"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(public.items())), encoding="utf-8")
        report_path = output_dir / "dependencies.json"
        write_json(report_path, {"status": "not_run"})
        auditor_home = (output_dir / "auditor-home").resolve()
        auditor_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (auditor_home / "AppData" / "Roaming").mkdir(parents=True, exist_ok=True, mode=0o700)
        (auditor_home / "AppData" / "Local").mkdir(parents=True, exist_ok=True, mode=0o700)
        process = subprocess.run([
            str(auditor_python.absolute()), "-I", "-B", "-m", "pip_audit", "-r", str(requirements.resolve()),
            "--no-deps", "--disable-pip", "--strict", "--format", "json", "--progress-spinner", "off",
            "--timeout", "30", "--cache-dir", str((output_dir / "http-cache").resolve()),
            "--output", str(report_path.resolve()),
        ], env=auditor_environment(auditor_home), capture_output=True, text=True, timeout=300, check=False)
        if process.returncode:
            raise AuditFailure("public_auditor_failed")
        validate_public_report(json.loads(report_path.read_text(encoding="utf-8")), public)
        summary = {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
                   "installed_count": len(inventory), "public_count": len(public),
                   "public_vulnerabilities": 0, "public_skips": 0,
                   "verified_private_count": 1, "private_version": private["version"],
                   "private_sqlite_version": private["sqlite_version"],
                   "python": sys.version.split()[0]}
        write_json(summary_path, summary)
        return summary
    except Exception as error:
        write_json(summary_path, {"status": "failed", "error_type": type(error).__name__,
                                  "reason_code": str(error) if isinstance(error, AuditFailure) else "audit_incomplete"})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/checks/security"))
    parser.add_argument("--sqlite-build-root", type=Path, default=builder.ROOT / "build")
    args = parser.parse_args()
    try:
        summary = run_audit(args.auditor_python, args.output_dir, args.sqlite_build_root)
    except Exception:
        print("Dependency audit failed; see audit-summary.json for the fixed diagnostic code.", file=sys.stderr)
        return 1
    print(f"Verified {summary['public_count']} public distributions without vulnerabilities or skips, "
          f"plus private SQLite {summary['private_sqlite_version']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

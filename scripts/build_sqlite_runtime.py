"""Build the private DB-API wheel against pinned, patched SQLite sources.

Run with the main2 virtual environment after installing requirements/native-build.lock.
No system Python or SQLite installation is modified. Install the resulting wheel
with pip; the public pysqlite3 0.6.0 wheel embeds an affected SQLite version.
"""

import argparse
import hashlib
from pathlib import Path
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

import certifi

ROOT = Path(__file__).resolve().parents[1]
SQLITE_VERSION = "3.53.4"
DRIVER_VERSION = "0.6.0+sqlite" + SQLITE_VERSION
SQLITE_ARCHIVE_SHA3_256 = "628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e"
SQLITE_C_SHA3_256 = "67f423e9ebbbdc473cbc4772c872ee6b89f31fde4ed0279a5c25d5f65c043a16"
SOURCES = (
    ("pysqlite3-0.6.0.tar.gz",
     "https://files.pythonhosted.org/packages/fa/84/6e586bef5f6337dee60066eef752c73fa6fcb93c1e7997b550d3105ed4f9/pysqlite3-0.6.0.tar.gz",
     "ecf5112b62a4e6c04438957e343fe9672707bd3191f789ecae6c95b226aa6bb6"),
    ("sqlite-amalgamation-3530400.zip", "https://sqlite.org/2026/sqlite-amalgamation-3530400.zip",
     "1e71ddf93849c6a6ecf58b827c0692073d2dd7ee40196158068f7b29f422e87d"),
)


def verified_source(directory: Path, name: str, url: str, expected: str, offline: bool) -> Path:
    path = directory / name
    if not path.exists():
        if offline:
            raise ValueError(f"Missing verified source: {name}")
        request = urllib.request.Request(url, headers={"User-Agent": "Trendanalyser-build/1.0"})
        # python.org macOS installs may not have system CA links configured.
        # Use the same pinned CA distribution as the application's HTTP client.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    return path


def build(offline: bool = False, *, build_root: Path | None = None) -> list[Path]:
    directory = (ROOT / "build" if build_root is None else build_root).resolve()
    sources = directory / "runtime-sources"
    archives = [verified_source(sources, *source, offline) for source in SOURCES]
    output = directory / "wheels"
    output.mkdir(parents=True, exist_ok=True)
    with archives[1].open("rb") as handle:
        if hashlib.file_digest(handle, "sha3_256").hexdigest() != SQLITE_ARCHIVE_SHA3_256:
            raise ValueError("SQLite archive differs from its officially published SHA3-256")
    with tempfile.TemporaryDirectory(prefix="sqlite-build-", dir=directory) as temporary:
        with tarfile.open(archives[0]) as archive:
            archive.extractall(temporary, filter="data")
        source_dir = Path(temporary) / "pysqlite3-0.6.0"
        with zipfile.ZipFile(archives[1]) as archive:
            for name in ("sqlite3.c", "sqlite3.h"):
                source = archive.read("sqlite-amalgamation-3530400/" + name)
                if name == "sqlite3.c" and hashlib.sha3_256(source).hexdigest() != SQLITE_C_SHA3_256:
                    raise ValueError("SQLite C source differs from its officially published SHA3-256")
                (source_dir / name).write_bytes(source)
        project = source_dir / "pyproject.toml"
        content = project.read_text(encoding="utf-8")
        # Distinguish our wheel from PyPI's wheel: pip must not silently substitute it.
        old = 'version = "0.6.0"'
        if content.count(old) != 1:
            raise ValueError("Unexpected driver version metadata")
        project.write_text(content.replace(old, f'version = "{DRIVER_VERSION}"'), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(output), str(source_dir)], check=True, timeout=300)
    wheels = sorted(output.glob(f"pysqlite3-{DRIVER_VERSION}-*.whl"))
    if not wheels:
        raise ValueError("Native wheel was not produced")
    return wheels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--build-root", type=Path, help="Separate source cache and wheel output; no environment is modified")
    args = parser.parse_args()
    for wheel in build(args.offline, build_root=args.build_root):
        print(wheel)


if __name__ == "__main__":
    main()

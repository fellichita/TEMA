"""Archived source bytes stay under the selected archive directory."""

import pytest

from app.backend.contracts import DocumentRecord
from app.pilot.archive import DocumentArchive
from app.pilot.library import write_artifact
from app.runtime.jobs import TaskFailure
from tests.platform_support import require_symlinks


def test_archive_root_symlink_is_rejected_before_any_revision_write(tmp_path):
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "revisions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TaskFailure, match="символической ссылкой"):
        DocumentArchive(tmp_path / "revisions")
    assert list(outside.iterdir()) == []


def test_revision_shard_symlink_cannot_redirect_document_write(tmp_path):
    require_symlinks()
    document = DocumentRecord(
        source="openalex", source_id="W1", title="A research paper",
        abstract="An independently observed result.", publication_year=2025,
        date_precision="year", url="https://openalex.org/W1",
    )
    archive = DocumentArchive(tmp_path / "revisions")
    outside = tmp_path / "outside"
    outside.mkdir()
    archive.directory.mkdir()
    import hashlib
    import json

    data = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    (archive.directory / digest[:2]).symlink_to(outside, target_is_directory=True)

    with pytest.raises(TaskFailure, match="символической ссылкой"):
        archive.put(document)
    assert list(outside.iterdir()) == []


def test_result_directory_symlink_cannot_redirect_artifact_write(tmp_path):
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(TaskFailure, match="символической ссылкой"):
        write_artifact(tmp_path / "results", {"result": "example"})
    assert list(outside.iterdir()) == []

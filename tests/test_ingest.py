# tests/test_ingest.py
# Unit tests for autobidsify/stages/ingest.py
# Tests cover: ingest_data with directory input (no-copy optimization),
#              ingest_data with ZIP archive, and ingest_info.json content.
# TAR extraction is also tested.

import json
import zipfile
import tarfile
import pytest
from pathlib import Path

from autobidsify.stages.ingest import ingest_data


# ============================================================================
# Directory input — no-copy optimization
# ============================================================================

class TestIngestDirectory:

    def test_creates_ingest_info(self, tmp_path):
        data_dir  = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scan.nii.gz").write_bytes(b"")
        out_dir   = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        assert (out_dir / "_staging" / "ingest_info.json").exists()

    def test_ingest_info_input_type_directory(self, tmp_path):
        data_dir  = tmp_path / "data"
        data_dir.mkdir()
        out_dir   = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["input_type"] == "directory"

    def test_ingest_info_actual_data_path_points_to_input(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        # actual_data_path must point to the original directory (no copying)
        assert Path(info["actual_data_path"]).resolve() == data_dir.resolve()

    def test_ingest_info_staging_dir_is_none_for_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["staging_dir"] is None

    def test_ingest_info_status_is_complete(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["status"] == "complete"

    def test_no_files_copied_for_directory_input(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scan.nii.gz").write_bytes(b"")
        out_dir = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        # Only _staging/ should exist in output — no data files copied
        output_files = list(out_dir.rglob("*.nii.gz"))
        assert len(output_files) == 0

    def test_staging_dir_created(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        assert (out_dir / "_staging").is_dir()


# ============================================================================
# ZIP archive input
# ============================================================================

class TestIngestZipArchive:

    def _make_zip(self, tmp_path: Path, files: dict) -> Path:
        """Create a ZIP with given {relpath: content} mapping."""
        zip_path = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for relpath, content in files.items():
                zf.writestr(relpath, content)
        return zip_path

    def test_zip_creates_ingest_info(self, tmp_path):
        zip_path = self._make_zip(tmp_path, {"scan.nii.gz": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(zip_path), out_dir)

        assert (out_dir / "_staging" / "ingest_info.json").exists()

    def test_zip_input_type_is_archive(self, tmp_path):
        zip_path = self._make_zip(tmp_path, {"scan.nii.gz": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(zip_path), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["input_type"] == "archive"

    def test_zip_files_extracted(self, tmp_path):
        zip_path = self._make_zip(tmp_path, {
            "sub-01/scan.nii.gz": "",
            "sub-01/scan.json":   "{}",
        })
        out_dir = tmp_path / "output"

        ingest_data(str(zip_path), out_dir)

        extracted = out_dir / "_staging" / "extracted"
        assert (extracted / "sub-01" / "scan.nii.gz").exists()
        assert (extracted / "sub-01" / "scan.json").exists()

    def test_zip_actual_data_path_is_extracted_dir(self, tmp_path):
        zip_path = self._make_zip(tmp_path, {"scan.dcm": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(zip_path), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert "extracted" in info["actual_data_path"]

    def test_zip_staging_dir_is_not_none(self, tmp_path):
        zip_path = self._make_zip(tmp_path, {"scan.dcm": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(zip_path), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["staging_dir"] is not None


# ============================================================================
# TAR archive input
# ============================================================================

class TestIngestTarArchive:

    def _make_tar_gz(self, tmp_path: Path, files: dict) -> Path:
        """Create a .tar.gz with given {relpath: content} mapping."""
        tar_path = tmp_path / "dataset.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for relpath, content in files.items():
                import io
                data = content.encode() if isinstance(content, str) else content
                info = tarfile.TarInfo(name=relpath)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return tar_path

    def test_tar_gz_files_extracted(self, tmp_path):
        tar_path = self._make_tar_gz(tmp_path, {"scan.nii.gz": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(tar_path), out_dir)

        extracted = out_dir / "_staging" / "extracted"
        assert (extracted / "scan.nii.gz").exists()

    def test_tar_gz_input_type_is_archive(self, tmp_path):
        tar_path = self._make_tar_gz(tmp_path, {"scan.dcm": ""})
        out_dir  = tmp_path / "output"

        ingest_data(str(tar_path), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["input_type"] == "archive"


# ============================================================================
# ingest_info.json content fields
# ============================================================================

class TestIngestInfoFields:

    def test_all_required_fields_present(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        for field in ["step", "timestamp", "input_path", "input_type",
                      "output_dir", "staging_dir", "actual_data_path", "status"]:
            assert field in info, f"Missing field: {field}"

    def test_step_is_ingest(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert info["step"] == "ingest"

    def test_timestamp_is_string(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir  = tmp_path / "output"

        ingest_data(str(data_dir), out_dir)

        info = json.loads((out_dir / "_staging" / "ingest_info.json").read_text())
        assert isinstance(info["timestamp"], str)
        assert len(info["timestamp"]) > 0

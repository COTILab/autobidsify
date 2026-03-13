# tests/test_utils.py
# Unit tests for autobidsify/utils.py
# Tests cover: file I/O, directory creation, JSON/YAML/text read-write,
#              file copy/tree, file listing, and hash utilities.

import json
import yaml
import pytest
from pathlib import Path

from autobidsify.utils import (
    ensure_dir,
    write_json, read_json,
    write_yaml, read_yaml,
    write_text, read_text,
    copy_file, copy_tree,
    list_all_files,
    sha1_head, sha256_full,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp(tmp_path):
    """Provide a clean temporary directory for each test."""
    return tmp_path


# ============================================================================
# ensure_dir
# ============================================================================

class TestEnsureDir:
    def test_creates_new_directory(self, tmp):
        target = tmp / "a" / "b" / "c"
        assert not target.exists()
        ensure_dir(target)
        assert target.is_dir()

    def test_idempotent_on_existing_directory(self, tmp):
        target = tmp / "existing"
        target.mkdir()
        # Should not raise
        ensure_dir(target)
        assert target.is_dir()

    def test_creates_deeply_nested_path(self, tmp):
        deep = tmp / "x" / "y" / "z" / "w"
        ensure_dir(deep)
        assert deep.is_dir()


# ============================================================================
# write_json / read_json
# ============================================================================

class TestJsonIO:
    def test_roundtrip_simple_dict(self, tmp):
        data = {"name": "test", "value": 42}
        path = tmp / "out.json"
        write_json(path, data)
        assert path.exists()
        result = read_json(path)
        assert result == data

    def test_roundtrip_nested_dict(self, tmp):
        data = {"a": {"b": [1, 2, 3]}, "c": None}
        path = tmp / "nested.json"
        write_json(path, data)
        assert read_json(path) == data

    def test_creates_parent_directories(self, tmp):
        path = tmp / "sub" / "dir" / "file.json"
        write_json(path, {"x": 1})
        assert path.exists()

    def test_unicode_content(self, tmp):
        data = {"label": "北京_sub001", "desc": "données"}
        path = tmp / "unicode.json"
        write_json(path, data)
        result = read_json(path)
        assert result["label"] == "北京_sub001"

    def test_written_file_is_valid_json(self, tmp):
        path = tmp / "valid.json"
        write_json(path, {"k": "v"})
        # Parse independently to confirm valid JSON
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed == {"k": "v"}

    def test_read_missing_file_raises(self, tmp):
        with pytest.raises(FileNotFoundError):
            read_json(tmp / "missing.json")


# ============================================================================
# write_yaml / read_yaml
# ============================================================================

class TestYamlIO:
    def test_roundtrip_simple_dict(self, tmp):
        data = {"subjects": ["sub-01", "sub-02"], "count": 2}
        path = tmp / "out.yaml"
        write_yaml(path, data)
        assert path.exists()
        result = read_yaml(path)
        assert result == data

    def test_creates_parent_directories(self, tmp):
        path = tmp / "nested" / "plan.yaml"
        write_yaml(path, {"key": "value"})
        assert path.exists()

    def test_written_file_is_valid_yaml(self, tmp):
        path = tmp / "valid.yaml"
        write_yaml(path, {"a": 1, "b": [2, 3]})
        with open(path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed["a"] == 1

    def test_none_value_roundtrip(self, tmp):
        data = {"key": None}
        path = tmp / "none.yaml"
        write_yaml(path, data)
        assert read_yaml(path) == data


# ============================================================================
# write_text / read_text
# ============================================================================

class TestTextIO:
    def test_roundtrip_plain_text(self, tmp):
        content = "Hello, BIDS!\nLine 2\n"
        path = tmp / "readme.txt"
        write_text(path, content)
        assert read_text(path) == content

    def test_creates_parent_directories(self, tmp):
        path = tmp / "a" / "b" / "readme.md"
        write_text(path, "# Title")
        assert path.exists()

    def test_unicode_content(self, tmp):
        content = "受试者\tparticipant\n"
        path = tmp / "uni.txt"
        write_text(path, content)
        assert read_text(path) == content

    def test_empty_string(self, tmp):
        path = tmp / "empty.txt"
        write_text(path, "")
        assert read_text(path) == ""


# ============================================================================
# copy_file
# ============================================================================

class TestCopyFile:
    def test_copies_file_to_new_location(self, tmp):
        src = tmp / "source.txt"
        src.write_text("content")
        dst = tmp / "subdir" / "dest.txt"
        copy_file(src, dst)
        assert dst.exists()
        assert dst.read_text() == "content"

    def test_creates_destination_parent(self, tmp):
        src = tmp / "file.json"
        src.write_text("{}")
        dst = tmp / "deep" / "path" / "file.json"
        copy_file(src, dst)
        assert dst.exists()

    def test_overwrites_existing_destination(self, tmp):
        src = tmp / "new.txt"
        src.write_text("new content")
        dst = tmp / "old.txt"
        dst.write_text("old content")
        copy_file(src, dst)
        assert dst.read_text() == "new content"


# ============================================================================
# copy_tree
# ============================================================================

class TestCopyTree:
    def test_copies_entire_directory(self, tmp):
        src = tmp / "src_tree"
        src.mkdir()
        (src / "a.txt").write_text("aaa")
        (src / "sub").mkdir()
        (src / "sub" / "b.txt").write_text("bbb")

        dst = tmp / "dst_tree"
        copy_tree(src, dst)

        assert (dst / "a.txt").read_text() == "aaa"
        assert (dst / "sub" / "b.txt").read_text() == "bbb"

    def test_merges_into_existing_destination(self, tmp):
        src = tmp / "src"
        src.mkdir()
        (src / "new.txt").write_text("new")

        dst = tmp / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("existing")

        copy_tree(src, dst)
        assert (dst / "new.txt").exists()
        assert (dst / "existing.txt").exists()


# ============================================================================
# list_all_files
# ============================================================================

class TestListAllFiles:
    def test_lists_files_recursively(self, tmp):
        (tmp / "a.txt").write_text("a")
        sub = tmp / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")
        (sub / "c.nii.gz").write_text("c")

        files = list_all_files(tmp)
        names = {f.name for f in files}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "c.nii.gz" in names

    def test_skips_hidden_files(self, tmp):
        (tmp / ".hidden").write_text("secret")
        (tmp / "visible.txt").write_text("visible")

        files = list_all_files(tmp)
        names = {f.name for f in files}
        assert ".hidden" not in names
        assert "visible.txt" in names

    def test_returns_only_files_not_dirs(self, tmp):
        (tmp / "file.txt").write_text("x")
        (tmp / "subdir").mkdir()

        files = list_all_files(tmp)
        for f in files:
            assert f.is_file()

    def test_empty_directory_returns_empty_list(self, tmp):
        empty = tmp / "empty"
        empty.mkdir()
        assert list_all_files(empty) == []


# ============================================================================
# sha1_head
# ============================================================================

class TestSha1Head:
    def test_returns_16_char_hex_string(self, tmp):
        path = tmp / "data.bin"
        path.write_bytes(b"hello world")
        result = sha1_head(path)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_files_have_different_hashes(self, tmp):
        p1 = tmp / "f1.bin"
        p2 = tmp / "f2.bin"
        p1.write_bytes(b"content A")
        p2.write_bytes(b"content B")
        assert sha1_head(p1) != sha1_head(p2)

    def test_same_content_same_hash(self, tmp):
        p1 = tmp / "copy1.bin"
        p2 = tmp / "copy2.bin"
        p1.write_bytes(b"identical")
        p2.write_bytes(b"identical")
        assert sha1_head(p1) == sha1_head(p2)

    def test_returns_error_string_for_missing_file(self, tmp):
        result = sha1_head(tmp / "nonexistent.bin")
        assert result == "error"


# ============================================================================
# sha256_full
# ============================================================================

class TestSha256Full:
    def test_returns_64_char_hex_string(self):
        result = sha256_full("test input")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        assert sha256_full("same") == sha256_full("same")

    def test_different_inputs_different_hashes(self):
        assert sha256_full("input A") != sha256_full("input B")

    def test_empty_string(self):
        result = sha256_full("")
        assert len(result) == 64

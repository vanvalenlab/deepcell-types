"""Unit tests for ``deepcell_types.utils._archive.extract_archive``.

These guard the security-relevant, network-free extraction paths: zip-slip /
tar-traversal / tar-symlink rejection and the member-count and size bounds.
"""

import io
import tarfile
import zipfile

import pytest

from deepcell_types.utils._archive import extract_archive


def test_extract_archive_accepts_benign_zip(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/ok.txt", "hello")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "inner" / "ok.txt").read_text() == "hello"


def test_extract_archive_rejects_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_accepts_benign_tar(tmp_path):
    archive = tmp_path / "ok.tar"
    data = b"hi"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("inner/ok.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "inner" / "ok.txt").read_bytes() == data


def test_extract_archive_rejects_tar_symlink_member(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_archive(archive, dest)


def test_extract_archive_rejects_tar_traversal_member(tmp_path):
    archive = tmp_path / "evil2.tar"
    data = b"x"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_archive(archive, dest)


def test_extract_archive_rejects_non_archive(tmp_path):
    plain = tmp_path / "notes.txt"
    plain.write_text("not an archive")
    with pytest.raises(ValueError, match="not a recognized"):
        extract_archive(plain, tmp_path / "out")


def test_extract_archive_enforces_member_and_size_limits(tmp_path):
    archive = tmp_path / "bounded.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one", b"123")
        zf.writestr("two", b"456")

    with pytest.raises(ValueError, match="2 members"):
        extract_archive(archive, tmp_path / "members", max_members=1)
    with pytest.raises(ValueError, match="declared size"):
        extract_archive(archive, tmp_path / "member-size", max_member_bytes=2)
    with pytest.raises(ValueError, match="larger than"):
        extract_archive(archive, tmp_path / "total-size", max_total_bytes=5)


def test_extract_archive_enforces_member_and_size_limits_tar(tmp_path):
    # The tar branch duplicates the zip branch's bound checks; exercise it
    # directly so a tar-only regression can't slip through.
    archive = tmp_path / "bounded.tar"
    with tarfile.open(archive, "w") as tf:
        for name, data in (("one", b"123"), ("two", b"456")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match="2 members"):
        extract_archive(archive, tmp_path / "members", max_members=1)
    with pytest.raises(ValueError, match="declared size"):
        extract_archive(archive, tmp_path / "member-size", max_member_bytes=2)
    with pytest.raises(ValueError, match="larger than"):
        extract_archive(archive, tmp_path / "total-size", max_total_bytes=5)



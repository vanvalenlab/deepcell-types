"""Unit tests for the safe archive extractor (``deepcell_types.utils._archive``):
zip-slip / tar-traversal / tar-symlink rejection and non-archive rejection.
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

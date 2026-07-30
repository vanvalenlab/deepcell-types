"""Path-traversal-safe extraction of downloaded archives.

Kept in this package rather than delegated to ``deepcell_auth``, whose
``extract_archive`` calls ``extractall`` without vetting members.
"""

import logging
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Deliberately generous limits accommodate the public multi-terabyte corpus
# while still bounding hostile or accidentally unbounded archives.
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_MEMBER_BYTES = 2 * 2**40
MAX_ARCHIVE_TOTAL_BYTES = 10 * 2**40


def extract_archive(
    archive_path,
    dest=None,
    *,
    max_members=MAX_ARCHIVE_MEMBERS,
    max_member_bytes=MAX_ARCHIVE_MEMBER_BYTES,
    max_total_bytes=MAX_ARCHIVE_TOTAL_BYTES,
):
    """Safely extract a ``.zip`` or ``.tar[.gz]`` archive.

    Rejects members whose resolved path would escape ``dest`` (zip-slip /
    tar path-traversal) and rejects tar symlink/hardlink members, so it is
    safe to call on archives downloaded from a remote source.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Path to a ``.zip`` or ``.tar``/``.tar.gz`` archive.
    dest : str or pathlib.Path, optional
        Destination directory. Defaults to the archive's parent directory.

    Returns
    -------
    pathlib.Path
        The destination directory the archive was extracted into.
    """
    archive_path = Path(archive_path)
    dest = Path(dest) if dest is not None else archive_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()

    def _within(name):
        try:
            (dest / name).resolve().relative_to(dest_resolved)
            return True
        except ValueError:
            return False

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            infos = zf.infolist()
            if len(infos) > max_members:
                raise ValueError(
                    f"Refusing to extract archive with {len(infos)} members; "
                    f"limit is {max_members}."
                )
            total = 0
            for member in infos:
                is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
                if not _within(member.filename):
                    raise ValueError(
                        f"Refusing to extract {member.filename!r}: path escapes {dest}."
                    )
                if is_symlink:
                    raise ValueError(
                        f"Refusing to extract unsafe zip member {member.filename!r}."
                    )
                if member.file_size > max_member_bytes:
                    raise ValueError(
                        f"Refusing to extract {member.filename!r}: declared size "
                        f"exceeds {max_member_bytes} bytes."
                    )
                total += member.file_size
                if total > max_total_bytes:
                    raise ValueError(
                        f"Refusing to extract archive larger than "
                        f"{max_total_bytes} bytes."
                    )
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            if len(members) > max_members:
                raise ValueError(
                    f"Refusing to extract archive with {len(members)} members; "
                    f"limit is {max_members}."
                )
            total = 0
            for member in members:
                if not (member.isfile() or member.isdir()) or not _within(member.name):
                    raise ValueError(
                        f"Refusing to extract unsafe tar member {member.name!r}."
                    )
                if member.size > max_member_bytes:
                    raise ValueError(
                        f"Refusing to extract {member.name!r}: declared size "
                        f"exceeds {max_member_bytes} bytes."
                    )
                total += member.size
                if total > max_total_bytes:
                    raise ValueError(
                        f"Refusing to extract archive larger than "
                        f"{max_total_bytes} bytes."
                    )
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"{archive_path} is not a recognized .zip or .tar archive.")

    logger.info(f"Extracted {archive_path} to {dest}")
    return dest

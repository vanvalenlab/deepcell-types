"""Safe archive extraction for downloaded assets.

Kept local (rather than delegated to ``deepcell-auth``) because
``deepcell_auth`` only downloads the training-data archive and its own
extractor uses an unguarded ``extractall``. This one rejects zip-slip /
tar path-traversal and tar symlink/hardlink members.
"""

import logging
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_archive(archive_path, dest=None):
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
            for member in zf.namelist():
                if not _within(member):
                    raise ValueError(
                        f"Refusing to extract {member!r}: path escapes {dest}."
                    )
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                if member.islnk() or member.issym() or not _within(member.name):
                    raise ValueError(
                        f"Refusing to extract unsafe tar member {member.name!r}."
                    )
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"{archive_path} is not a recognized .zip or .tar archive.")

    logger.info(f"Extracted {archive_path} to {dest}")
    return dest

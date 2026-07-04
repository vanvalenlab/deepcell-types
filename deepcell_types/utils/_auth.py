"""User interface to authentication layer for data/models."""

import os
import tarfile
import zipfile
import requests
from pathlib import Path
from hashlib import md5, sha256
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


_api_endpoint = "https://users.deepcell.org/api/getData/"
_asset_location = Path.home() / ".deepcell"

# Hash algorithm is selected by digest length so existing md5-pinned assets keep
# working while new assets can pin the stronger sha256. SHA-256 is preferred for
# new entries (md5 is collision-weak).
_HASH_BY_HEXLEN = {32: ("md5", md5), 64: ("sha256", sha256)}


def _hash_file(fpath, file_hash):
    """Return ``(algo_name, hexdigest)`` for ``fpath`` using the algorithm
    implied by the length of ``file_hash`` (32 hex → md5, 64 hex → sha256).

    Hashes in chunks so multi-GB assets are not read fully into memory.
    """
    try:
        algo_name, algo = _HASH_BY_HEXLEN[len(file_hash)]
    except KeyError:
        raise ValueError(
            f"Unrecognized file_hash length {len(file_hash)}; expected an md5 "
            "(32 hex chars) or sha256 (64 hex chars) digest."
        )
    hasher = algo()
    with open(fpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return algo_name, hasher.hexdigest()


def fetch_data(asset_key: str, cache_subdir=None, file_hash=None):
    """Fetch assets through users.deepcell.org authentication system.

    Download assets from the deepcell suite of datasets and models which
    require user-authentication.

    .. note::

       You must have a Deepcell Access Token set as an environment variable
       with the name ``DEEPCELL_ACCESS_TOKEN`` in order to access assets.

       Access tokens can be created at <https://users.deepcell.org>_

    Args:
        :param asset_key: Key of the file to download.
        The list of available assets can be found on the users.deepcell.org
        homepage.

        :param cache_subdir: `str` indicating directory relative to
        `~/.deepcell` where downloaded data will be cached. The default is
        `None`, which means cache the data in `~/.deepcell`.

        :param file_hash: `str` representing the md5 (32 hex chars) or sha256
        (64 hex chars) checksum of datafile; the algorithm is auto-detected from
        the length. The checksum is used to perform data caching. If no checksum
        is provided or the checksum differs from that found in the data cache,
        the data will be (re)-downloaded.
    """
    download_location = _asset_location
    if cache_subdir is not None:
        download_location /= cache_subdir
    download_location.mkdir(exist_ok=True, parents=True)

    # Extract the filename from the asset_key, which can be a full path
    fname = os.path.split(asset_key)[-1]
    fpath = download_location / fname

    # Check for cached data
    if file_hash is not None:
        logger.info("Checking for cached data")
        try:
            logger.info(f"Checking {fname} against provided file_hash...")
            _, digest = _hash_file(fpath, file_hash)
            if digest == file_hash:
                logger.info(f"{fname} with hash {file_hash} already available.")
                return fpath
            logger.info(
                f"{fname} with hash {file_hash} not found in {download_location}"
            )
        except FileNotFoundError:
            pass

    # Check for access token
    access_token = os.environ.get("DEEPCELL_ACCESS_TOKEN")
    if access_token is None:
        raise ValueError(
            "\nDEEPCELL_ACCESS_TOKEN not found.\n"
            "Please set your access token to the DEEPCELL_ACCESS_TOKEN\n"
            "environment variable.\n"
            "For example:\n\n"
            "\texport DEEPCELL_ACCESS_TOKEN=<your-token>.\n\n"
            "If you don't yet have a token, you can create one at\n"
            "https://users.deepcell.org"
        )

    # Request download URL
    headers = {"X-Api-Key": access_token}
    logger.info("Making request to server")
    resp = requests.post(
        _api_endpoint,
        headers=headers,
        data={"s3_key": asset_key},
        timeout=(30, 300),
    )

    def _safe_json(r):
        # Gateways/proxies often return HTML or empty error bodies; don't let a
        # JSONDecodeError mask the real HTTP status.
        try:
            return r.json()
        except ValueError:
            return {}

    # Raise informative exception for the specific case when the asset_key is
    # not found in the bucket
    if resp.status_code == 404 and _safe_json(resp).get("error") == "Key not found":
        raise ValueError(f"Object {asset_key} not found.")
    # Raise informative exception for the specific case when an invalid
    # API token is provided.
    if resp.status_code == 403 and (
        _safe_json(resp).get("detail")
        == "Authentication credentials were not provided."
    ):
        raise ValueError(
            "\n\nThe provided DEEPCELL_ACCESS_TOKEN is not valid.\n"
            "The token may be expired - if so, create a new one at\n"
            "https://users.deepcell.org"
        )
    # Handle all other non-http-200 status
    resp.raise_for_status()

    # Parse response
    response_data = _safe_json(resp)
    if "url" not in response_data:
        raise ValueError(
            f"Unexpected response from {_api_endpoint} (status {resp.status_code}): "
            f"missing download URL. Body starts: {resp.text[:200]!r}"
        )
    download_url = response_data["url"]
    file_size = response_data.get("size")
    # The server-side ``size`` field comes back as a string like "12.3 MB"; parse
    # it into bytes for the progress bar, but fall back to an unknown total
    # (``None``) rather than aborting an otherwise-valid download if the format
    # is unexpected.
    suffix_mapping = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    try:
        val, suff = str(file_size).split(" ")
        file_size_numerical = int(float(val) * suffix_mapping[suff.upper()])
    except (ValueError, KeyError, AttributeError):
        file_size_numerical = None

    logger.info(f"Downloading {asset_key} with size {file_size} to {download_location}")
    data_req = requests.get(
        download_url,
        headers={"user-agent": "Wget/1.20 (linux-gnu)"},
        stream=True,
        timeout=(30, 300),
    )
    data_req.raise_for_status()

    chunk_size = 4096
    try:
        with tqdm.wrapattr(
            open(fpath, "wb"), "write", miniters=1, total=file_size_numerical
        ) as fh:
            for chunk in data_req.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
    except BaseException:
        # Don't leave a half-written file on disk: subsequent runs (especially
        # for callers that don't pass file_hash) would silently return the
        # corrupt path. Drop and re-raise.
        if fpath.exists():
            try:
                fpath.unlink()
            except OSError:
                pass
        raise

    # Verify the downloaded file against the expected hash, if one was given.
    # This catches truncated downloads that completed without raising and
    # protects against tampered intermediaries.
    if file_hash is not None:
        algo_name, actual = _hash_file(fpath, file_hash)
        if actual != file_hash:
            fpath.unlink(missing_ok=True)
            raise ValueError(
                f"Integrity check failed for {fname}: "
                f"expected {algo_name}={file_hash}, got {algo_name}={actual}. "
                "The downloaded file has been removed; please retry."
            )

    logger.info(f"Successfully downloaded {fname} to {fpath}")

    return fpath


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

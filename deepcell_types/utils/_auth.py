"""User interface to authentication layer for data/models."""

import os
import requests
from pathlib import Path
from hashlib import md5
from tqdm import tqdm
import logging


_api_endpoint = "https://users.deepcell.org/api/getData/"
_asset_location = Path.home() / ".deepcell"


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

        :param file_hash: `str` represented the md5 checksum of datafile. The
        checksum is used to perform data caching. If no checksum is provided or
        the checksum differs from that found in the data cache, the data will
        be (re)-downloaded.
    """
    logging.basicConfig(level=logging.INFO)

    download_location = _asset_location
    if cache_subdir is not None:
        download_location /= cache_subdir
    download_location.mkdir(exist_ok=True, parents=True)

    # Extract the filename from the asset_key, which can be a full path
    fname = os.path.split(asset_key)[-1]
    fpath = download_location / fname

    # Check for cached data
    if file_hash is not None:
        logging.info('Checking for cached data')
        try:
            with open(fpath, "rb") as fh:
                hasher = md5(fh.read())
            logging.info(f"Checking {fname} against provided file_hash...")
            md5sum = hasher.hexdigest()
            if md5sum == file_hash:
                logging.info(
                    f"{fname} with hash {file_hash} already available."
                )
                return fpath
            logging.info(
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
    logging.info("Making request to server")
    resp = requests.post(
        _api_endpoint, headers=headers, data={"s3_key": asset_key}
    )
    # Raise informative exception for the specific case when the asset_key is
    # not found in the bucket
    if resp.status_code == 404 and resp.json().get("error") == "Key not found":
        raise ValueError(f"Object {asset_key} not found.")
    # Raise informative exception for the specific case when an invalid
    # API token is provided.
    if resp.status_code == 403 and (
       resp.json().get("detail") == "Authentication credentials were not provided."
    ):
        raise ValueError(
            "\n\nThe provided DEEPCELL_ACCESS_TOKEN is not valid.\n"
            "The token may be expired - if so, create a new one at\n"
            "https://users.deepcell.org"
        )
    # Handle all other non-http-200 status
    resp.raise_for_status()

    # Parse response
    response_data = resp.json()
    download_url = response_data["url"]
    file_size = response_data["size"]
    # The server-side ``size`` field comes back as a string like "12.3 MB"; parse
    # it into bytes so the progress bar shows a real total.
    val, suff = file_size.split(" ")
    suffix_mapping = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    file_size_numerical = int(float(val) * suffix_mapping[suff])

    logging.info(
        f"Downloading {asset_key} with size {file_size} to {download_location}"
    )
    data_req = requests.get(
        download_url, headers={"user-agent": "Wget/1.20 (linux-gnu)"}, stream=True
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
        with open(fpath, "rb") as fh:
            actual = md5(fh.read()).hexdigest()
        if actual != file_hash:
            fpath.unlink(missing_ok=True)
            raise ValueError(
                f"Integrity check failed for {fname}: "
                f"expected md5={file_hash}, got md5={actual}. "
                "The downloaded file has been removed; please retry."
            )

    logging.info(f"Successfully downloaded {fname} to {fpath}")

    return fpath

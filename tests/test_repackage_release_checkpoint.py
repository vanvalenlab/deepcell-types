"""Unit tests for the checkpoint re-packaging helper.

``scripts/repackage_release_checkpoint.bundle_vocabulary`` back-fills the
``ct2idx`` + ``canonical_channels`` keys that ``validate_checkpoint_vocabulary``
requires, sourcing them from a :class:`DCTConfig`. These tests pin its contract
without touching the filesystem or any real asset: it bundles the config's
vocabulary, and refuses inputs where doing so would be unsafe (a mismatched
class count, an already-bundled checkpoint, or a non-checkpoint dict).
"""

import pytest

from deepcell_types.config import DCTConfig
from scripts.repackage_release_checkpoint import bundle_vocabulary


@pytest.fixture(scope="module")
def cfg():
    # Packaged vocab.json (archive-free); no network or archive needed.
    return DCTConfig()


def _fake_ckpt(n_celltypes):
    return {"model": {}, "config": {"n_celltypes": n_celltypes}}


def test_bundles_config_vocabulary(cfg):
    ckpt = _fake_ckpt(len(cfg.ct2idx))
    bundle_vocabulary(ckpt, cfg)
    assert ckpt["ct2idx"] == dict(cfg.ct2idx)
    assert ckpt["canonical_channels"] == list(cfg.marker2idx.keys())
    # A bundled checkpoint now satisfies the ordering guard against the same cfg.
    from deepcell_types.predict import validate_checkpoint_vocabulary

    validate_checkpoint_vocabulary(ckpt, cfg.ct2idx, cfg.marker2idx)


def test_refuses_mismatched_n_celltypes(cfg):
    ckpt = _fake_ckpt(len(cfg.ct2idx) + 1)
    with pytest.raises(ValueError, match="mismatched vocabulary"):
        bundle_vocabulary(ckpt, cfg)


def test_refuses_already_bundled(cfg):
    ckpt = _fake_ckpt(len(cfg.ct2idx))
    ckpt["ct2idx"] = dict(cfg.ct2idx)
    with pytest.raises(ValueError, match="already bundles"):
        bundle_vocabulary(ckpt, cfg)


def test_refuses_non_checkpoint_dict(cfg):
    with pytest.raises(ValueError, match="'model' and 'config'"):
        bundle_vocabulary({"config": {"n_celltypes": len(cfg.ct2idx)}}, cfg)

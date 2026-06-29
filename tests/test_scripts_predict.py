"""Unit tests for the evaluation CLI's marker-embedding resolution
(``scripts/predict.py``). These exercise the path that lets the canonical
evaluation run without an ``svd_512.npz`` file: when ``--svd_embeddings_path``
is omitted, a correctly-shaped zeros placeholder is built from the checkpoint
(its values are overwritten by ``load_state_dict``). No archive or network is
needed — the helper is pure.
"""

import numpy as np
import pytest

from scripts.predict import _resolve_marker_embeddings


class _Cfg:
    """Minimal stand-in for ``TissueNetConfig`` exposing only what the helper
    touches: ``NUM_MARKERS`` and the SVD loader."""

    NUM_MARKERS = 3

    def __init__(self, svd_return=None):
        self._svd_return = svd_return

    def load_marker_embeddings_array(self, svd_path):
        assert svd_path == "some/path.npz"
        return self._svd_return


def test_resolve_builds_zeros_placeholder_when_no_svd_path():
    state_dict = {
        "marker_embedder.embed_layer.weight": np.zeros((4, 8), dtype=np.float32)
    }
    out = _resolve_marker_embeddings(_Cfg(), state_dict, None)
    # Shape is (NUM_MARKERS, embed_dim); embed_dim read from the checkpoint.
    assert out.shape == (3, 8)
    assert out.dtype == np.float32
    # Placeholder: values are overwritten by load_state_dict, so zeros is fine.
    assert not out.any()


def test_resolve_errors_when_embedding_dim_unknown():
    with pytest.raises(ValueError, match="marker-embedding dimension"):
        _resolve_marker_embeddings(_Cfg(), {}, None)


def test_resolve_delegates_to_svd_loader_when_path_given():
    sentinel = np.ones((3, 5), dtype=np.float32)
    out = _resolve_marker_embeddings(_Cfg(svd_return=sentinel), {}, "some/path.npz")
    assert out is sentinel

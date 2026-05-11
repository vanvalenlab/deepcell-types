"""Regression tests for load_marker_embeddings_array alignment (issue #14, gap 5).

The historical `embed_dim=0 -> Linear(0, 256)` all-zero-output bug was exactly
this class of silent misalignment. The current code validates `marker2idx`
from the npz and reindexes to the archive's `marker2idx`, but there is no
test that:
  (a) a matching-name npz uses the fast path,
  (b) a re-ordered npz is reindexed correctly,
  (c) a missing-marker npz logs a warning and zero-fills the gap.

Reference: deepcelltypes/config.py::TissueNetConfig.load_marker_embeddings_array
"""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from deepcell_types.training.config import TissueNetConfig


def _make_stub_config(marker2idx):
    """Return an object that exposes the attributes
    `load_marker_embeddings_array` reads (`marker2idx`, `NUM_MARKERS`) without
    invoking the heavy zarr-backed constructor.
    """
    stub = SimpleNamespace()
    stub.marker2idx = marker2idx
    stub.NUM_MARKERS = len(marker2idx)
    # Bind the method as unbound and call it with the stub as `self`
    stub.load_marker_embeddings_array = (
        lambda svd_path, embedding_model_name="deepseek-r1-70b":
        TissueNetConfig.load_marker_embeddings_array(
            stub,  # type: ignore[arg-type]
            embedding_model_name=embedding_model_name,
            svd_path=svd_path,
        )
    )
    return stub


class TestLoadMarkerEmbeddingsArray:
    """Canonical marker alignment: archive order is the ground truth."""

    def _archive_marker2idx(self):
        return {"CD3": 0, "CD4": 1, "CD8": 2, "CD45": 3}

    def test_matching_marker2idx_fast_path(self, tmp_path):
        """Identical saved and archive marker2idx: embeddings returned as-is."""
        m2i = self._archive_marker2idx()
        # Each row is a distinct signature to confirm the right row ends up in
        # each slot.
        embeds = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        npz_path = tmp_path / "matching.npz"
        np.savez(
            npz_path,
            marker_embeddings=embeds,
            marker2idx=np.array(m2i, dtype=object),
        )

        stub = _make_stub_config(m2i)
        out = stub.load_marker_embeddings_array(str(npz_path))

        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out, embeds)

    def test_reordered_marker2idx_reindexes(self, tmp_path):
        """Saved npz has a permuted marker2idx; output must align with archive order."""
        archive_m2i = self._archive_marker2idx()
        # Saved order: CD45, CD8, CD4, CD3 (reverse of archive)
        saved_m2i = {"CD45": 0, "CD8": 1, "CD4": 2, "CD3": 3}
        saved_embeds = np.array(
            [
                [40.0, 40.0, 40.0],  # CD45 at row 0
                [80.0, 80.0, 80.0],  # CD8 at row 1
                [40.0, 40.0, 40.0],  # CD4 at row 2 (same as CD45 by accident in v1 suite is fine)
                [30.0, 30.0, 30.0],  # CD3 at row 3
            ],
            dtype=np.float32,
        )
        # Use distinct signatures so a bug shows up loudly.
        saved_embeds[0] = [100.0, 45.0, 0.0]  # CD45
        saved_embeds[1] = [200.0, 8.0,  0.0]  # CD8
        saved_embeds[2] = [300.0, 4.0,  0.0]  # CD4
        saved_embeds[3] = [400.0, 3.0,  0.0]  # CD3

        npz_path = tmp_path / "reordered.npz"
        np.savez(
            npz_path,
            marker_embeddings=saved_embeds,
            marker2idx=np.array(saved_m2i, dtype=object),
        )

        stub = _make_stub_config(archive_m2i)
        out = stub.load_marker_embeddings_array(str(npz_path))

        # Archive order is CD3, CD4, CD8, CD45 -> rows must come from
        # saved rows 3, 2, 1, 0
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out[0], saved_embeds[3])  # CD3
        np.testing.assert_array_equal(out[1], saved_embeds[2])  # CD4
        np.testing.assert_array_equal(out[2], saved_embeds[1])  # CD8
        np.testing.assert_array_equal(out[3], saved_embeds[0])  # CD45

    def test_missing_marker_zero_filled_and_warns(self, tmp_path, caplog):
        """A marker present in the archive but missing from the saved npz is
        zero-filled AND logged as a warning. Silent zero-fill (no warning)
        would reproduce the original embed_dim=0 class of bug.
        """
        archive_m2i = self._archive_marker2idx()
        # Saved npz is missing 'CD45' entirely (only 3 markers).
        saved_m2i = {"CD3": 0, "CD4": 1, "CD8": 2}
        saved_embeds = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        npz_path = tmp_path / "missing.npz"
        np.savez(
            npz_path,
            marker_embeddings=saved_embeds,
            marker2idx=np.array(saved_m2i, dtype=object),
        )

        stub = _make_stub_config(archive_m2i)

        with caplog.at_level(logging.WARNING, logger="deepcell_types.training.config"):
            out = stub.load_marker_embeddings_array(str(npz_path))

        assert out.shape == (4, 3)
        # Known markers aligned by name
        np.testing.assert_array_equal(out[0], saved_embeds[0])  # CD3
        np.testing.assert_array_equal(out[1], saved_embeds[1])  # CD4
        np.testing.assert_array_equal(out[2], saved_embeds[2])  # CD8
        # Missing marker CD45 is zero-filled (NOT random, NOT garbage from an
        # unrelated row).
        np.testing.assert_array_equal(out[3], np.zeros(3, dtype=np.float32))

        # A warning must have been logged mentioning the missing marker.
        missing_warnings = [
            rec for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and "missing" in rec.getMessage().lower()
        ]
        assert len(missing_warnings) >= 1, (
            "Expected a warning when markers are missing from saved marker2idx"
        )

    def test_missing_svd_path_raises(self, tmp_path):
        """Non-existent svd_path raises FileNotFoundError with a clear message."""
        stub = _make_stub_config(self._archive_marker2idx())
        with pytest.raises(FileNotFoundError):
            stub.load_marker_embeddings_array(str(tmp_path / "does-not-exist.npz"))

    def test_no_svd_path_raises(self):
        """Calling without svd_path raises ValueError (no silent default)."""
        stub = _make_stub_config(self._archive_marker2idx())
        with pytest.raises(ValueError):
            stub.load_marker_embeddings_array(None)

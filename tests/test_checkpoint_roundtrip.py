"""Regression test for checkpoint save/load + atomic-write cleanup.

`scripts/train.py` writes checkpoints via `torch.save({"model": ...}, tmp_path)`
followed by `os.replace(tmp_path, final_path)` for atomicity. Tests that:
(a) the saved file round-trips into a fresh model with matching state_dict,
(b) `torch.load(..., weights_only=True)` succeeds on the payload,
(c) atomic-write leaves no `.tmp` file behind on successful save.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn


class _TinyNet(nn.Module):
    """Minimal two-layer model used as a checkpoint stand-in."""

    def __init__(self, d_in=4, d_hidden=8, d_out=3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _atomic_save(obj, final_path: Path):
    """Mirror the atomic-save idiom used in scripts/train.py."""
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)
    return tmp_path


class TestCheckpointRoundTrip:
    def test_state_dict_round_trip(self, tmp_path):
        """A saved {"model": state_dict()} must restore exactly into a fresh model."""
        torch.manual_seed(0)
        original = _TinyNet()
        ckpt_path = tmp_path / "model.pt"
        _atomic_save({"model": original.state_dict()}, ckpt_path)

        # Load into a fresh instance
        loaded_model = _TinyNet()
        # Sanity: before loading, weights differ (different init)
        torch.manual_seed(1)
        loaded_model = _TinyNet()
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_model.load_state_dict(payload["model"])

        # Every parameter tensor must match bitwise
        orig_sd = original.state_dict()
        load_sd = loaded_model.state_dict()
        assert set(orig_sd.keys()) == set(load_sd.keys())
        for k in orig_sd:
            assert torch.equal(orig_sd[k], load_sd[k]), f"Mismatch on {k}"

    def test_weights_only_true_succeeds(self, tmp_path):
        """weights_only=True must succeed on the checkpoint payload.

        This guards against a future refactor that adds a non-tensor/unsafe
        object to the checkpoint dict (e.g. a lambda or a custom class not
        registered as safe), which would start raising
        `torch.serialization.UnpicklingError` under weights_only=True.
        """
        original = _TinyNet()
        ckpt_path = tmp_path / "model.pt"
        _atomic_save({"model": original.state_dict()}, ckpt_path)

        # Should NOT raise
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model" in payload

    def test_no_tmp_file_after_success(self, tmp_path):
        """Atomic save must not leave a .tmp sibling behind on success."""
        original = _TinyNet()
        ckpt_path = tmp_path / "model.pt"
        tmp_path_dot_tmp = _atomic_save({"model": original.state_dict()}, ckpt_path)

        assert ckpt_path.exists(), "Final file must exist"
        assert not tmp_path_dot_tmp.exists(), (
            f"Atomic save leaked tmp file: {tmp_path_dot_tmp}"
        )
        # More exhaustive: no .tmp siblings anywhere in the directory
        stray_tmps = list(tmp_path.glob("*.tmp"))
        assert stray_tmps == [], f"Stray .tmp files: {stray_tmps}"

    def test_atomic_save_is_overwrite(self, tmp_path):
        """Second atomic save overwrites the first; no .tmp lingers."""
        ckpt_path = tmp_path / "model.pt"
        net1 = _TinyNet()
        _atomic_save({"model": net1.state_dict()}, ckpt_path)

        # Save a differently-sized payload (extra metadata)
        net2 = _TinyNet(d_hidden=16)  # larger model -> larger state dict
        _atomic_save({"model": net2.state_dict(), "epoch": 5}, ckpt_path)

        assert ckpt_path.exists()
        assert list(tmp_path.glob("*.tmp")) == []
        # The second save must have actually replaced the file
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert payload.get("epoch") == 5
        # And the new model's dict shapes match net2, not net1
        assert payload["model"]["fc1.weight"].shape == (16, 4)

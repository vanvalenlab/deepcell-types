"""Smoke test: post-v7 R&D features (arcsinh, rank, jitter, QN, TTA BN) are stripped from HEAD."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _help_output(script_module: str) -> str:
    result = subprocess.run(
        ["python", "-m", script_module, "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return result.stdout + result.stderr

def test_train_flags_removed():
    out = _help_output("scripts.train")
    for flag in ("--arcsinh_cofactor", "--use_rank_features", "--intensity_jitter", "--quantile_ref_path"):
        assert flag not in out, f"{flag} still in scripts/train.py --help"

def test_predict_flags_removed():
    out = _help_output("scripts.predict")
    for flag in ("--arcsinh_cofactor", "--use_rank_features", "--tta_bn"):
        assert flag not in out, f"{flag} still in scripts/predict.py --help"

def test_dataset_no_rank_params():
    import inspect
    from deepcell_types.training.dataset import FullImageDataset, create_dataloader
    ds_params = inspect.signature(FullImageDataset.__init__).parameters
    for p in ("arcsinh_cofactor", "use_rank_features", "quantile_ref_path"):
        assert p not in ds_params, f"{p} still a FullImageDataset init kwarg"
    dl_params = inspect.signature(create_dataloader).parameters
    for p in ("arcsinh_cofactor", "use_rank_features", "quantile_ref_path"):
        assert p not in dl_params, f"{p} still a create_dataloader kwarg"

def test_no_intensity_jitter_class():
    from deepcell_types.training import dataset
    assert not hasattr(dataset, "IntensityJitter"), "IntensityJitter class still present"

def test_batchdata_no_channel_ranks():
    from deepcell_types.training.utils import BatchData
    import dataclasses
    fields = {f.name for f in dataclasses.fields(BatchData)}
    assert "channel_ranks" not in fields, "channel_ranks still in BatchData"

def test_model_no_use_rank_features():
    import inspect
    from deepcell_types.model import create_model
    assert "use_rank_features" not in inspect.signature(create_model).parameters

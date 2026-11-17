"""Hand-derived golden test for cellsighter's pure tensor helper.

convert_batch_for_cellsighter() scatters per-dataset channels to their global
marker positions and appends the cell + neighbor masks. The two synthetic cases
below have hand-derived outputs (cross-checked against the original function).

Layout: B=1, C_max=3, H=W=1, num_markers=4. sample[:, :, 0, 0, 0] = channel values.

Case 1 (no index collision):
  values=[2,3,5], ch_idx=[1,3,-1], mask=[F,F,T] (channel 2 is padding)
  -> masked values [2,3,0]; scatter to global [1,3,(0 via clamp, src 0)]
  -> global_patches = [0, 2, 0, 3]; append cell=0.7, neighbor=0.2
  -> [0, 2, 0, 3, 0.7, 0.2]

Case 2 (documents the scatter index-0 clobber quirk; CPU last-write-wins):
  values=[2,3,5], ch_idx=[0,3,-1], mask=[F,F,T]
  -> masked [2,3,0]; clamped idx [0,3,0]; channel0 writes 2 to global0, then the
     padded channel2 writes 0 to global0 LAST -> global0 clobbered to 0
  -> global_patches = [0, 0, 0, 3]; append cell=0.1, neighbor=0.9
  -> [0, 0, 0, 3, 0.1, 0.9]
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "torchvision"
)  # cellsighter.model imports torchvision at module load


def _make_batch(values, ch_idx, mask, cell, neigh, H=1, W=1):
    from deepcell_types.training.utils import BatchData

    C = len(values)
    sample = (
        torch.tensor(values, dtype=torch.float32)
        .reshape(1, C, 1, 1, 1)
        .expand(1, C, 1, H, W)
        .clone()
    )
    spatial = torch.zeros(1, 3, H, W)
    spatial[0, 0] = cell
    spatial[0, 1] = neigh
    kw = {}
    for f in dataclasses.fields(BatchData):
        kw[f.name] = {
            "sample": sample,
            "spatial_context": spatial,
            "ch_idx": torch.tensor([ch_idx], dtype=torch.long),
            "mask": torch.tensor([mask], dtype=torch.bool),
            "marker_positivity_mask": torch.ones(1, C, dtype=torch.bool),
        }.get(f.name, None)
    return BatchData(**kw)


def test_convert_batch_no_collision_hand_derived():
    from deepcell_types.baselines.cellsighter.model import convert_batch_for_cellsighter

    bd = _make_batch([2.0, 3.0, 5.0], [1, 3, -1], [False, False, True], 0.7, 0.2)
    out = convert_batch_for_cellsighter(bd, num_markers=4)
    assert tuple(out.shape) == (1, 6, 1, 1)  # num_markers + 2
    assert out.reshape(-1).tolist() == pytest.approx([0.0, 2.0, 0.0, 3.0, 0.7, 0.2])


def test_convert_batch_index0_clobber_quirk_preserved():
    from deepcell_types.baselines.cellsighter.model import convert_batch_for_cellsighter

    # A real marker at global index 0 is clobbered to 0 by a padded channel
    # (scatter_ duplicate-index last-write-wins on CPU). Pinned, not fixed.
    bd = _make_batch([2.0, 3.0, 5.0], [0, 3, -1], [False, False, True], 0.1, 0.9)
    out = convert_batch_for_cellsighter(bd, num_markers=4)
    assert out.reshape(-1).tolist() == pytest.approx([0.0, 0.0, 0.0, 3.0, 0.1, 0.9])

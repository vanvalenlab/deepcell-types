"""Characterization test for nimbus's pure metric reducer.

compute_marker_positivity_metrics() is the only importable pure helper in the
nimbus baseline (load_fov_data needs zarr; _predict_with_tta is an inner closure
in main()). The synthetic case below has a hand-derived confusion matrix:

  threshold = 0.5
  GT (per cell_type): Tcell -> CD3+ CD20- ; Bcell -> CD3- CD20+
  4 cells:
    cell0 Tcell  CD3=0.9 CD20=0.1   -> CD3:TP  CD20:TN
    cell1 Tcell  CD3=0.4 CD20=0.2   -> CD3:FN  CD20:TN
    cell2 Bcell  CD3=0.1 CD20=0.8   -> CD3:TN  CD20:TP
    cell3 Bcell  CD3=0.2 CD20=0.3   -> CD3:TN  CD20:FN

  Per marker (CD3 and CD20 identical): tp=1 fp=0 fn=1 tn=2
    accuracy=0.75 precision=1.0 recall=0.5 f1=2/3  n_samples=4
  Global pool (8 rows): tp=2 fp=0 fn=2 tn=4
    accuracy=0.75 precision=1.0 recall=0.5 f1=2/3  n_samples=8
"""

import pandas as pd
import pytest

from deepcell_types.baselines.nimbus.run import compute_marker_positivity_metrics


def _make_inputs():
    gt_df = pd.DataFrame(
        {"CD3": {"Tcell": 1, "Bcell": 0}, "CD20": {"Tcell": 0, "Bcell": 1}}
    )  # index = cell types, columns = markers
    ground_truth = {"ds1": gt_df}
    predictions = pd.DataFrame(
        {
            "cell_index": [0, 1, 2, 3],
            "cell_type": ["Tcell", "Tcell", "Bcell", "Bcell"],
            "dataset_name": ["ds1", "ds1", "ds1", "ds1"],
            "CD3": [0.9, 0.4, 0.1, 0.2],
            "CD20": [0.1, 0.2, 0.8, 0.3],
        }
    )
    return predictions, ground_truth


def test_overall_metrics_hand_derived():
    predictions, ground_truth = _make_inputs()
    out = compute_marker_positivity_metrics(predictions, ground_truth, threshold=0.5)
    o = out["overall"]
    assert o["n_samples"] == 8
    assert o["accuracy"] == pytest.approx(0.75)
    assert o["precision"] == pytest.approx(1.0)
    assert o["recall"] == pytest.approx(0.5)
    assert o["f1"] == pytest.approx(2 / 3)
    # mp_* reduction values are fully determined by the confusion matrix above.
    # macro = mean over the two identical markers; micro pools all 8 rows.
    assert o["mp_macro_f1"] == pytest.approx(2 / 3)
    assert o["mp_micro_f1"] == pytest.approx(2 / 3)
    assert o["mp_macro_precision"] == pytest.approx(1.0)
    assert o["mp_macro_recall"] == pytest.approx(0.5)
    assert o["mp_macro_accuracy"] == pytest.approx(0.75)
    assert o["mp_micro_precision"] == pytest.approx(1.0)
    assert o["mp_micro_recall"] == pytest.approx(0.5)
    assert o["mp_num_markers"] == 2


def test_per_marker_metrics_hand_derived():
    predictions, ground_truth = _make_inputs()
    out = compute_marker_positivity_metrics(predictions, ground_truth, threshold=0.5)
    for marker in ("CD3", "CD20"):
        m = out["per_marker"][marker]
        assert m["n_samples"] == 4
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(2 / 3)

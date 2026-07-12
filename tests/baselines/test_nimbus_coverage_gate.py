"""Coverage-accounting test for the Nimbus baseline's FOV-loading loop.

``check_dataset_coverage`` is the pure helper extracted from ``main()`` (the
loop itself needs a real/mocked zarr archive + Nimbus model, so isn't
directly unit-testable). It must always report the skipped/failed dataset
count, and raise ``RuntimeError`` only once the failure rate exceeds the 1%
threshold that mirrors ``training/dataset.py``'s cell-data cache build gate
(see ``deepcell_types/training/dataset.py``, ``failed_keys`` / ``fail_rate``).
Baselines are scored at full coverage (``deepcell_types/abstention.py``), so
a silent drop must never be invisible — hence the aggregate print happens
unconditionally, even below threshold.
"""

import pytest

from deepcell_types.baselines.nimbus.run import (
    NIMBUS_DATASET_FAILURE_RATE_THRESHOLD,
    check_dataset_coverage,
)


def test_no_skips_reports_zero_and_does_not_raise(capsys):
    check_dataset_coverage(skipped_keys=[], total_keys=100)
    out = capsys.readouterr().out
    assert "Skipped 0 of 100 datasets" in out


def test_below_threshold_reports_but_does_not_raise(capsys):
    # 1 of 100 == 1.0%, at (not above) the default 1% threshold -> no raise.
    check_dataset_coverage(skipped_keys=["ds_bad"], total_keys=100)
    out = capsys.readouterr().out
    assert "Skipped 1 of 100 datasets" in out


def test_above_threshold_raises_runtime_error(capsys):
    skipped = [f"ds_bad_{i}" for i in range(5)]
    with pytest.raises(RuntimeError, match="above the 1% safety threshold"):
        check_dataset_coverage(skipped_keys=skipped, total_keys=100)
    # The aggregate line is printed before the raise, so it's never silent.
    out = capsys.readouterr().out
    assert "Skipped 5 of 100 datasets" in out


def test_custom_threshold_is_respected(capsys):
    # 2 of 100 == 2%, above a tightened 1.5% threshold.
    with pytest.raises(RuntimeError, match="above the 2% safety threshold"):
        check_dataset_coverage(
            skipped_keys=["a", "b"], total_keys=100, threshold=0.015
        )


def test_default_threshold_constant_matches_convention():
    # Pins the module constant to the training/dataset.py cache-build
    # convention (>1% dataset drop is a schema regression, not noise).
    assert NIMBUS_DATASET_FAILURE_RATE_THRESHOLD == 0.01

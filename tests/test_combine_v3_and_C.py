"""Unit tests for scripts/_combine_v3_and_C.py.

Three required cases:
  (a) mutual-exclusion fires on a synthetic both-POS case
  (b) implication rule does NOT downgrade when intensity is in mid-range
      (i.e., not below 0.5 * threshold)
  (c) implication rule DOES downgrade when intensity is below the lower
      fence (intensity < 0.5 * threshold)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


# Allow `import scripts._combine_v3_and_C` regardless of cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def combine_module():
    return importlib.import_module("scripts._combine_v3_and_C")


# ---------------------------------------------------------------------------
# (a) mutual exclusion: both POS -> both "?"
# ---------------------------------------------------------------------------

def test_mutual_exclusion_fires_on_both_pos(combine_module):
    """CD3+ and CD19+ on the same cell -> abstain on both."""
    labels = {"CD3": 1, "CD19": 1, "CD45": 1}
    intensities = {"CD3": 100.0, "CD19": 100.0, "CD45": 200.0}
    thresholds = {"CD3": 50.0, "CD19": 50.0, "CD45": 50.0}
    counts: dict = {}
    out = combine_module.apply_rules_to_cell(
        labels, intensities, thresholds, rule_counts=counts,
    )
    assert out["CD3"] == "?", f"CD3 should be masked, got {out['CD3']}"
    assert out["CD19"] == "?", f"CD19 should be masked, got {out['CD19']}"
    # CD45 untouched (no rule fires for it here besides the implication
    # rules, which require CD3+/CD20+/CD19+/CD68+ -- but those calls are
    # now "?" after exclusion, so implication should not fire on this cell)
    assert out["CD45"] == 1, f"CD45 should be unchanged, got {out['CD45']}"
    # exactly one mutual-exclusion fired
    excl_total = sum(v for k, v in counts.items() if k.startswith("excl["))
    assert excl_total == 1, f"expected 1 exclusion fire, got {counts}"


# ---------------------------------------------------------------------------
# (b) implication does NOT fire when intensity is mid-range (>= 0.5 * thr)
# ---------------------------------------------------------------------------

def test_implication_does_not_fire_for_mid_range_intensity(combine_module):
    """CD3+ (POS), CD45 binary call=0 BUT continuous intensity is mid-range
    (e.g. 30 with threshold 50, so > 0.5 * 50 = 25) -> CD3 stays POS.

    This is the design fix vs B+C: in B+C, CD45=0 (binary) would have
    triggered downgrade; here the continuous intensity (30) is NOT below
    the lower fence (25), so the rule abstains from firing.
    """
    labels = {"CD3": 1, "CD45": 0}
    # CD45 mean intensity = 30, threshold = 50; 30 >= 0.5 * 50 = 25.
    # NOT strong-NEG, so implication should NOT fire.
    intensities = {"CD3": 80.0, "CD45": 30.0}
    thresholds = {"CD3": 50.0, "CD45": 50.0}
    counts: dict = {}
    out = combine_module.apply_rules_to_cell(
        labels, intensities, thresholds, rule_counts=counts,
    )
    assert out["CD3"] == 1, (
        f"CD3 should remain POS (CD45 intensity 30 not strong-NEG vs "
        f"threshold 50 with factor 0.5), got {out['CD3']}"
    )
    assert out["CD45"] == 0, f"CD45 should remain NEG, got {out['CD45']}"
    impl_total = sum(v for k, v in counts.items() if k.startswith("impl["))
    assert impl_total == 0, f"no implication should fire, got {counts}"


# ---------------------------------------------------------------------------
# (c) implication DOES fire when intensity is below the lower fence
# ---------------------------------------------------------------------------

def test_implication_fires_when_intensity_below_lower_fence(combine_module):
    """CD3+ (POS), CD45 intensity BELOW the strong-NEG cutoff
    (< 0.5 * threshold) -> CD3 downgraded to "?"."""
    labels = {"CD3": 1, "CD45": 0}
    # CD45 mean intensity = 5.0, threshold = 50.0; 5.0 < 0.5 * 50 = 25.
    # Strong-NEG -> CD3 should be downgraded.
    intensities = {"CD3": 80.0, "CD45": 5.0}
    thresholds = {"CD3": 50.0, "CD45": 50.0}
    counts: dict = {}
    out = combine_module.apply_rules_to_cell(
        labels, intensities, thresholds, rule_counts=counts,
    )
    assert out["CD3"] == "?", (
        f"CD3 should be downgraded (CD45 intensity 5 < 0.5*50=25), "
        f"got {out['CD3']}"
    )
    assert out["CD45"] == 0, f"CD45 should remain NEG, got {out['CD45']}"
    impl_total = sum(v for k, v in counts.items() if k.startswith("impl["))
    assert impl_total == 1, f"exactly one implication should fire, got {counts}"


# ---------------------------------------------------------------------------
# Bonus sanity checks (non-required but cheap)
# ---------------------------------------------------------------------------

def test_no_rule_fires_when_all_negative(combine_module):
    labels = {"CD3": 0, "CD19": 0, "CD45": 0}
    intensities = {"CD3": 1.0, "CD19": 1.0, "CD45": 1.0}
    thresholds = {"CD3": 50.0, "CD19": 50.0, "CD45": 50.0}
    counts: dict = {}
    out = combine_module.apply_rules_to_cell(
        labels, intensities, thresholds, rule_counts=counts,
    )
    assert out == labels
    assert sum(counts.values()) == 0


def test_marker_canonicalization(combine_module):
    """Aliases like 'aSMA' should resolve to canonical 'SMA' under the rules."""
    # SMA / CD45 mutual exclusion: both POS -> both "?"
    labels = {"aSMA": 1, "CD45": 1}
    intensities = {"SMA": 100.0, "CD45": 100.0}
    thresholds = {"SMA": 50.0, "CD45": 50.0}
    out = combine_module.apply_rules_to_cell(
        labels, intensities, thresholds,
    )
    assert out["aSMA"] == "?"
    assert out["CD45"] == "?"

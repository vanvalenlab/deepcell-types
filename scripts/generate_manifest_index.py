"""Generate machine-auditable INDEX.tsv files for models/ and output/.

Run via: ``python scripts/generate_manifest_index.py``

The generator is idempotent: running it twice produces byte-identical TSVs
(no runtime timestamps embedded). Each row lists one file with:

    filename<TAB>size_bytes<TAB>size_human<TAB>mtime_iso<TAB>category<TAB>canonical

- ``filename`` is the bare filename for ``models/`` (flat directory) and
  ``<subdir>/<file>`` (POSIX) for ``output/`` (which has subdirs).
- ``canonical = yes`` flags the 8-ish files enumerated in the MANIFEST's
  canonical-checkpoint / canonical-prediction tables; everything else is
  ``no``. Keeps future MANIFEST drift visible via ``git diff``.

Not part of the canonical training pipeline — this is a throw-away helper.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "output"


# Canonical files referenced in models/MANIFEST.md and output/MANIFEST.md.
# Keep in sync with those tables. `maps_v7_2` is canonical per
# docs/reports/experiment_scores.md (80.9%/88.7% beats maps_v7_1 at 79.4%/89.1%).
CANONICAL_MODELS: frozenset[str] = frozenset(
    {
        "model_exp_v7_resnet48_0_best.pt",
        "model_exp_v8_lora8_2_best.pt",
        "model_exp_v8_tumor_head_0_best.pt",
        "cellsighter_cellsighter_v7_0.pth",
        "maps_maps_v7_2.pth",
        "maps_maps_v7_2_stats.npz",
        "model_pretrain_v7_0_best.pt",
        "xgb_model_xgb_v7_1.json",
    }
)


# Canonical output artifacts: every file whose basename begins with one of
# these stems is marked canonical. The stems mirror the MANIFEST's
# "Canonical" table verbatim.
CANONICAL_OUTPUT_STEMS: tuple[str, ...] = (
    "exp_v7_resnet48_0",
    "exp_v8_lora8_2",
    "exp_v8_tumor_head_0",
    "exp_v8_seen_mp_all_datasets",
    "cellsighter_v7_0",
    "maps_v7_2",
    "xgb_v7_1",
    "nimbus_gold_",
    "nimbus_val_v7_",
    "nimbus_all_mp_",
    "nimbus_split_",
    "gold_standard_ours",
    "gold_standard_nimbus",
)


def human_size(n: int) -> str:
    """Return a short human-readable size string (B, K, M, G, T).

    Matches ``du -h`` style: integer for < 10, one decimal otherwise.
    """
    step = 1024.0
    units = ["B", "K", "M", "G", "T", "P"]
    size = float(n)
    idx = 0
    while size >= step and idx < len(units) - 1:
        size /= step
        idx += 1
    if idx == 0:
        return f"{int(size)}{units[idx]}"
    if size >= 10:
        return f"{int(round(size))}{units[idx]}"
    return f"{size:.1f}{units[idx]}"


def mtime_iso(ts: float) -> str:
    """ISO 8601 timestamp (UTC, seconds precision) for a POSIX mtime."""
    return (
        _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Models category classifier
# ---------------------------------------------------------------------------


def classify_model(name: str) -> str:
    """Return the category bucket for a ``models/`` filename."""
    # Order matters: holdout check before v-prefix buckets (holdout filenames
    # can contain ``_skip_``), and ``pretrain`` before ``exp_v7`` (there is no
    # ``model_exp_v7_pretrain`` family, but be defensive).
    if name.startswith("model_pretrain_"):
        return "pretrain"
    if name.startswith("model_") and (
        "_skip_" in name or "_holdout" in name or name.startswith("model_final_run_")
    ):
        return "holdout_archived"
    if name.startswith("model_exp_v7_"):
        return "v7"
    if name.startswith("model_exp_v8_"):
        return "v8"
    if name.startswith("model_exp_v9_"):
        return "v9_archived"
    if name.startswith("model_exp_v10_"):
        return "v10_archived"
    if name.startswith("model_exp_v5_") or name.startswith("model_exp_v6_"):
        return "v5_v6_legacy"
    if name.startswith("model_exp_v2_") or name.startswith("model_exp_v3_"):
        return "v2_v3_legacy"
    if name.startswith("model_v2_") or name.startswith("model_v3_"):
        # Pre-"exp_" naming convention (model_v2_fix_*, model_v2_update_*, ...).
        return "v2_v3_legacy"
    if name.startswith("cellsighter_"):
        return "cellsighter"
    if name.startswith("maps_"):
        return "maps"
    if name.startswith("xgb_model_") or name.startswith("xgb_tuned_"):
        return "xgboost"
    if name.startswith("nimbus_"):
        return "nimbus"
    # Transformer-architecture R&D families described in models/MANIFEST.md
    # lines 87-97 ("Transformer architecture experiments, ~400 files").
    if (
        name.startswith("model_c_patch2_")
        or name.startswith("model_patch2_")
        or name.startswith("model_longer_")
        or name.startswith("model_entropic_")
        or name.startswith("model_clip_")
        or name.startswith("model_noisy_clip_")
        or name.startswith("model_nonclip_")
        or name.startswith("model_run_")
        or name.startswith("model_run_znormed_")
        or name.startswith("model_test_")
        or name.startswith("model_test_by_fov")
        or name.startswith("model_single_attn")
        or name.startswith("model_archive_model_")
        or name.startswith("model_base_skip_")
    ):
        return "architecture_archived"
    # Batch 3/4/5 one-off ablations named ``model_exp_<change>_0_best.pt``
    # (bn_lrfix, tcell, gamma1, hierarch, warmup15, ...). See MANIFEST line 97.
    if name.startswith("model_exp_") and name.endswith(".pt"):
        return "ablation_archived"
    return "other"


# ---------------------------------------------------------------------------
# Output category classifier
# ---------------------------------------------------------------------------

_EXP_PREFIX_RE = re.compile(r"^(exp_v(\d+))_")


def classify_output(rel_path: str) -> str:
    """Return the category bucket for an ``output/`` relative path."""
    name = Path(rel_path).name

    # Subdirectory artifacts: keep them bucketed by the top-level dir name
    # (e.g. "hybrid_contextual_full_gpt41/..." stays one bucket).
    parts = Path(rel_path).parts
    if len(parts) > 1:
        head = parts[0]
        if head == "archive":
            return "archive"
        if head == "__pycache__":
            return "pycache"
        if head == "figures":
            return "figures_dir"
        if head == "tuning":
            return "tuning_dir"
        if head.startswith("hybrid"):
            return "hybrid_dir"
        if head.startswith("llm_phenotyper"):
            return "llm_phenotyper_dir"
        if head == "binary_holdout_full":
            return "holdout_archived"
        if head == "output_marker_pos":
            return "marker_pos_dir"
        return f"subdir_{head}"

    # Flat files: first key off extension-based buckets, then experiment prefix.
    suffix = Path(name).suffix.lower()
    if suffix == ".png":
        return "figure"
    if suffix == ".log":
        return "log"

    # Experiment-prefixed predictions / metrics / logs / NPZs.
    m = _EXP_PREFIX_RE.match(name)
    if m:
        major = int(m.group(2))
        if major in (2, 3):
            return "v2_v3_legacy"
        if major in (5, 6):
            return "v5_v6_legacy"
        if major == 7:
            return "v7"
        if major == 8:
            return "v8"
        if major == 9:
            return "v9_archived"
        if major == 10:
            return "v10_archived"

    if name.startswith("cellsighter_"):
        return "cellsighter"
    if name.startswith("maps_"):
        return "maps"
    if name.startswith("xgb_") or name.startswith("xgboost_"):
        return "xgboost"
    if name.startswith("nimbus_"):
        return "nimbus"
    if name.startswith("gold_standard_"):
        return "gold_standard"
    if name.startswith("final_run_") or "_skip_" in name or "holdout" in name:
        return "holdout_archived"
    if name.startswith("Liu_"):
        return "liu_per_tissue"
    if name.startswith("c_patch2_") or name.startswith("patch2_"):
        return "patch2_archived"
    if name.startswith("longer_") or name.startswith("entropic_"):
        return "architecture_archived"
    if name.startswith("clip_") or name.startswith("noisy_clip_") or name.startswith("nonclip_"):
        return "architecture_archived"
    if name.startswith("run_") or name.startswith("test_") or name.startswith("cls_token_"):
        return "architecture_archived"
    if name.startswith("v2_"):
        return "v2_v3_legacy"

    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "npz"
    if suffix == ".csv":
        return "csv"
    if suffix == ".md":
        return "manifest"
    if suffix == ".py":
        return "script"
    return "other"


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def build_rows_models() -> list[tuple]:
    rows: list[tuple] = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in {"MANIFEST.md", "INDEX.tsv"}:
            continue
        st = entry.stat()
        canonical = "yes" if entry.name in CANONICAL_MODELS else "no"
        rows.append(
            (
                entry.name,
                st.st_size,
                human_size(st.st_size),
                mtime_iso(st.st_mtime),
                classify_model(entry.name),
                canonical,
            )
        )
    return rows


def is_canonical_output(name: str) -> bool:
    return any(name.startswith(stem) for stem in CANONICAL_OUTPUT_STEMS)


def build_rows_output() -> list[tuple]:
    rows: list[tuple] = []
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(OUTPUT_DIR)
        rel_str = rel.as_posix()
        if rel_str in {"MANIFEST.md", "INDEX.tsv"}:
            continue
        if rel.parts and rel.parts[0] == "__pycache__":
            # Always record them under pycache bucket; keep for completeness.
            pass
        st = path.stat()
        basename = path.name
        # An artifact is canonical if its basename matches one of the canonical
        # prediction-family stems. Subdirectory artifacts are archived (not
        # canonical) by definition since MANIFEST lists no canonical subdirs.
        canonical = "yes" if (len(rel.parts) == 1 and is_canonical_output(basename)) else "no"
        rows.append(
            (
                rel_str,
                st.st_size,
                human_size(st.st_size),
                mtime_iso(st.st_mtime),
                classify_output(rel_str),
                canonical,
            )
        )
    return rows


def sort_rows(rows: list[tuple]) -> list[tuple]:
    """Canonical rows first (alphabetical), then by (category, filename)."""
    canonical = sorted([r for r in rows if r[5] == "yes"], key=lambda r: r[0])
    other = sorted([r for r in rows if r[5] != "yes"], key=lambda r: (r[4], r[0]))
    return canonical + other


HEADER = (
    "# filename\tsize_bytes\tsize_human\tmtime_iso\tcategory\tcanonical\n"
    "# Generated by scripts/generate_manifest_index.py. Sort order: canonical=yes first (alpha), then by (category, filename).\n"
)


def write_tsv(path: Path, rows: list[tuple]) -> None:
    lines = [HEADER]
    for row in rows:
        lines.append("\t".join(str(c) for c in row) + "\n")
    path.write_text("".join(lines))


def summarize(label: str, rows: list[tuple]) -> None:
    total = len(rows)
    n_canon = sum(1 for r in rows if r[5] == "yes")
    n_other = sum(1 for r in rows if r[4] == "other")
    print(f"{label}: {total} rows, {n_canon} canonical, {n_other} fall-through (category=other)")


def main() -> None:
    model_rows = sort_rows(build_rows_models())
    output_rows = sort_rows(build_rows_output())
    write_tsv(MODELS_DIR / "INDEX.tsv", model_rows)
    write_tsv(OUTPUT_DIR / "INDEX.tsv", output_rows)
    summarize("models/INDEX.tsv", model_rows)
    summarize("output/INDEX.tsv", output_rows)


if __name__ == "__main__":
    main()

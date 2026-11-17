"""Cell-type hierarchy used for hierarchical evaluation.

Predictions of child types count as correct when the ground-truth label is a
parent type; the training loss still uses exact labels. Extracted from
``config.py`` so the mapping has a single, import-light home (re-exported from
``deepcell_types.training.config`` for backward compatibility).
"""

CELL_TYPE_HIERARCHY = {
    "Tcell": ["CD4T", "CD8T", "Treg", "NKT"],
    "Stromal": ["Fibroblast", "Pericyte"],
}

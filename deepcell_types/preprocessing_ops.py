"""Declarative, bounded per-FOV preprocessing ops for the ``preprocess`` hook.

Each config is a list of ``{"op": name, ...params}`` dicts applied in order to a
``(C, H, W)`` float32 array; the result must be in ``[0, 1]`` (end with
``min_max_normalize``). Build a hook for ``deepcell_types.predict`` with
``make_preprocessor(config)``.

``DEFAULT_CONFIG`` reproduces the built-in inference preprocessing
(per-channel nonzero-pixel p99 clip + min-max), so passing it is equivalent to
passing no hook at all.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# Built-in inference percentile (mirrors DCTConfig.PERCENTILE_THRESHOLD).
_DEFAULT_PERCENTILE = 99.0

DEFAULT_CONFIG = [
    {"op": "clip_percentile", "p": _DEFAULT_PERCENTILE},
    {"op": "min_max_normalize"},
]


def _clip_percentile_nonzero(x, p):
    """Per-channel clip at the p-th percentile of *nonzero* pixels.

    Matches ``preprocessing._percentile_threshold_nonzero`` so ``DEFAULT_CONFIG``
    is bit-equivalent to the built-in path. All-zero channels are untouched.
    """
    out = x.copy()
    for c in range(x.shape[0]):
        nz = out[c][np.nonzero(out[c])]
        if nz.size:
            hi = np.percentile(nz, p)
            np.minimum(out[c], hi, out=out[c])
    return out


def _min_max(x):
    mn = x.min(axis=(1, 2), keepdims=True)
    ptp = np.ptp(x, axis=(1, 2), keepdims=True)
    ptp[ptp == 0] = 1.0
    return (x - mn) / ptp


def _hot_pixel_removal(x, z):
    out = x.copy()
    for c in range(x.shape[0]):
        med = np.median(out[c])
        mad = np.median(np.abs(out[c] - med)) or 1.0
        hi = med + z * 1.4826 * mad
        np.minimum(out[c], hi, out=out[c])
    return out


def apply_config(raw, channel_names, config):
    """Apply a bounded op pipeline to ``raw`` (C,H,W) -> (C,H,W) float32.

    ``channel_names`` are the resolved standard marker names aligned to ``raw``'s
    channels (what the ``preprocess`` hook receives from ``predict``).
    """
    x = np.asarray(raw, dtype=np.float32).copy()
    idx = {n: i for i, n in enumerate(channel_names)}
    for step in config:
        op = step["op"]
        if op == "clip_percentile":
            x = _clip_percentile_nonzero(x, float(step["p"]))
        elif op == "arcsinh":
            x = np.arcsinh(x / float(step.get("cofactor", 5.0)))
        elif op == "log1p":
            x = np.log1p(np.clip(x, 0, None))
        elif op == "background_subtract":
            x = np.clip(x - float(step["value"]), 0, None)
        elif op == "gamma":
            mx = x.max(axis=(1, 2), keepdims=True)
            mx[mx == 0] = 1.0
            x = np.power(np.clip(x / mx, 0, 1), float(step["g"])) * mx
        elif op == "denoise":
            kind = step.get("kind", "median")
            size = int(step.get("size", 3))
            for c in range(x.shape[0]):
                if kind == "median":
                    x[c] = median_filter(x[c], size=size)
                elif kind == "gaussian":
                    x[c] = gaussian_filter(x[c], sigma=size / 2.0)
                else:
                    raise ValueError(f"unknown denoise kind {kind!r}")
        elif op == "hot_pixel_removal":
            x = _hot_pixel_removal(x, float(step.get("z", 5.0)))
        elif op == "channel_drop":
            for n in step["names"]:
                if n in idx:
                    x[idx[n]] = 0.0
        elif op == "channel_weight":
            for n, w in step["weights"].items():
                if n in idx:
                    x[idx[n]] *= float(w)
        elif op == "min_max_normalize":
            x = _min_max(x)
        else:
            raise ValueError(f"unknown op {op!r}")
    return x.astype(np.float32)


def make_preprocessor(config) -> Callable[[np.ndarray, List[str]], np.ndarray]:
    """Return a ``preprocess`` hook for ``deepcell_types.predict`` from a config."""
    return lambda raw, channel_names: apply_config(raw, channel_names, config)

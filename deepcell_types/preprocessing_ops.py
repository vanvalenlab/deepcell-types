"""Declarative, bounded per-FOV preprocessing ops for the ``preprocess`` hook.

Each config is a list of ``{"op": name, ...params}`` dicts applied in order to a
``(C, H, W)`` float32 array; the result must be in ``[0, 1]`` (end with
``min_max_normalize``). Build a hook for ``deepcell_types.predict`` with
``make_preprocessor(config)``.

``DEFAULT_CONFIG`` reproduces the built-in inference preprocessing
(per-channel nonzero-pixel p99.9 clip + min-max), so passing it is equivalent to
passing no hook at all.

Note: ``DEFAULT_CONFIG`` (a preprocessing op list) is unrelated to
:class:`deepcell_types.DCTConfig` (the inference marker / cell-type registry),
despite the similar name.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

__all__ = ["apply_config", "make_preprocessor", "DEFAULT_CONFIG"]

# Built-in inference percentile (mirrors DCTConfig.PERCENTILE_THRESHOLD).
_DEFAULT_PERCENTILE = 99.9

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
    ptp = x.max(axis=(1, 2), keepdims=True) - mn
    ptp[ptp == 0] = 1.0
    return (x - mn) / ptp


def _background_subtract_per_channel(x, p, channels=None):
    """Per-channel subtract the p-th percentile of *nonzero* pixels (the channel's
    own background floor), clipped at 0.

    Unlike ``background_subtract`` (one global value for every channel), this
    removes each channel's own pedestal: a high-background channel (e.g. a poorly
    normalized CD15 sitting on a large floor) gets a large subtraction while a
    clean channel whose floor is already near zero is barely touched. Restrict to
    specific channels with ``channels`` (indices); ``None`` applies to all.
    """
    out = x.copy()
    sel = range(x.shape[0]) if channels is None else channels
    for c in sel:
        nz = out[c][np.nonzero(out[c])]
        if nz.size:
            floor = np.percentile(nz, p)
            out[c] = np.clip(out[c] - floor, 0, None)
    return out


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

    def require_known(names):
        unknown = sorted(set(names) - set(idx))
        if unknown:
            raise ValueError(
                f"unknown preprocessing channel name(s): {unknown}; "
                f"available channels: {sorted(idx)}"
            )

    for step in config:
        op = step["op"]
        if op == "clip_percentile":
            x = _clip_percentile_nonzero(x, float(step["p"]))
        elif op == "log1p":
            x = np.log1p(np.clip(x, 0, None))
        elif op == "background_subtract":
            x = np.clip(x - float(step["value"]), 0, None)
        elif op == "background_subtract_per_channel":
            names = step.get("names")
            if names:
                require_known(names)
            sel = [idx[n] for n in names] if names else None
            x = _background_subtract_per_channel(x, float(step.get("p", 25.0)), sel)
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
            require_known(step["names"])
            for n in step["names"]:
                x[idx[n]] = 0.0
        elif op == "channel_weight":
            require_known(step["weights"])
            for n, w in step["weights"].items():
                x[idx[n]] *= float(w)
        elif op == "min_max_normalize":
            x = _min_max(x)
        else:
            raise ValueError(f"unknown op {op!r}")
    return x.astype(np.float32)


def make_preprocessor(config) -> Callable[[np.ndarray, List[str]], np.ndarray]:
    """Return a ``preprocess`` hook for ``deepcell_types.predict`` from a config."""
    return lambda raw, channel_names: apply_config(raw, channel_names, config)

#!/usr/bin/env python
"""
Fold MarkerEmbeddingLayer LoRA weights into the trainable projection.

Old architecture (removed):
    out = proj(raw_emb) + lora_B(lora_A(raw_emb))
Equivalent:
    out = (proj.W + lora_B.W @ lora_A.W) @ raw_emb + proj.b

This script reads a checkpoint with ``marker_embedder.lora_A.weight`` and
``marker_embedder.lora_B.weight``, folds them into ``marker_embedder.proj.weight``,
deletes the LoRA keys, and writes a new checkpoint that's loadable by the
current LoRA-free codebase.

Usage:
    python scripts/fold_lora_into_proj.py <in_ckpt> <out_ckpt>
    # or in place:
    python scripts/fold_lora_into_proj.py <ckpt>
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch


def fold(ckpt: dict) -> tuple[dict, bool]:
    """Fold LoRA into proj for every MarkerEmbeddingLayer in the state dict.

    Handles both the main CT-side embedder (``marker_embedder.*``) and the
    shared MP-head embedder (``marker_pos_head.marker_embedding_layer.*``).
    Each has its own lora_A/lora_B keys in the saved state_dict even though
    the underlying nn.Module is typically shared.
    """
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    changed = False
    prefixes = sorted({
        k.rsplit(".lora_A.weight", 1)[0]
        for k in sd
        if k.endswith(".lora_A.weight") and k.replace(".lora_A.weight", ".proj.weight") in sd
    })
    for prefix in prefixes:
        A = sd.pop(f"{prefix}.lora_A.weight")             # (rank, embed_dim)
        B = sd.pop(f"{prefix}.lora_B.weight")             # (d_model, rank)
        delta = B @ A                                     # (d_model, embed_dim)
        proj_key = f"{prefix}.proj.weight"
        sd[proj_key] = sd[proj_key] + delta
        print(f"  folded {prefix}.lora_{{A,B}} → {proj_key} (delta norm {delta.norm():.4f})")
        changed = True
    return ckpt, changed


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src
    print(f"Loading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt, changed = fold(ckpt)
    if not changed:
        print("No LoRA keys found, nothing to fold")
        if src != dst:
            torch.save(ckpt, dst)
            print(f"Copied unchanged to {dst}")
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(dst)
    print(f"Wrote {dst} (LoRA folded into marker_embedder.proj)")


if __name__ == "__main__":
    main()

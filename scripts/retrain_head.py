"""Canonical stage-2: retrain the cell-type head on the FROZEN backbone.

The default DeepCellTypes recipe is two-stage and *decoupled*:
  stage 1 (train.py)   – train the backbone with the weighted sampler on, so rare
                         classes get enough exposure to learn good features.
  stage 2 (this script) – freeze the backbone, train a residual-MLP head on the
                         NATURAL class distribution (sampler OFF, plain CE).

Why decouple: the sqrt-inverse-frequency WeightedRandomSampler helps the backbone
but *hurts* the head — it over-fires rare classes at the expense of the common,
well-supported classes that dominate macro-F1. Training the head sampler-free on
the frozen representation lifts full-coverage macro-F1 from ~70.8 to ~79.1, beating
the XGBoost baseline (76.6). End-to-end sampler-off training instead *erodes* the
backbone, so the two stages must stay separate.

Output: a deployable checkpoint whose config records the residual-MLP head
architecture and dimensions (so `predict.py` reconstructs the head), with the
feature standardization folded into the head's first layer so it consumes the
raw CLS embedding.

Usage:
  python scripts/retrain_head.py --pretrained_path models/model_<backbone>_best.pt \
      --svd_embeddings_path <svd.npz> --split_file <split.json> \
      --output models/model_<name>_resmlp_best.pt
"""

import os
import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from deepcell_types.training.config import TissueNetConfig
from deepcell_types.training.dataset import create_dataloader
from deepcell_types.model import create_model, ResidualMLPHead
from deepcell_types.training.utils import BatchData, seed_everything

DATA_DIR = os.environ.get("DATA_DIR", "")


def _extract_cls(model, loader, device, desc):
    """Forward the frozen backbone; return (cls embeddings, labels)."""
    cls_all, lab_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            bd = BatchData(*batch).to(device)
            out = model(
                bd.sample,
                bd.spatial_context,
                bd.ch_idx,
                bd.mask,
                return_attn_weights=False,
                domain_idx=bd.domain_idx,
            )
            cls_all.append(out.cls_embedding.half().cpu().numpy())
            lab_all.append(bd.ct_idx.cpu().numpy().astype(np.int64))
    return np.concatenate(cls_all), np.concatenate(lab_all)


@click.command()
@click.option("--pretrained_path", required=True, help="Stage-1 backbone checkpoint")
@click.option("--svd_embeddings_path", required=True)
@click.option("--split_file", required=True)
@click.option("--zarr_dir", default=DATA_DIR)
@click.option(
    "--output", required=True, help="Path for the deployable resMLP checkpoint"
)
@click.option("--device_num", default="cuda:0")
@click.option("--batch_size", type=int, default=512)
@click.option("--num_workers", type=int, default=12)
@click.option("--epochs", type=int, default=50)
@click.option("--lr", type=float, default=1.5e-3)
@click.option("--width", type=int, default=512)
@click.option("--depth", type=int, default=4)
@click.option("--seed", type=int, default=42)
def main(
    pretrained_path,
    svd_embeddings_path,
    split_file,
    zarr_dir,
    output,
    device_num,
    batch_size,
    num_workers,
    epochs,
    lr,
    width,
    depth,
    seed,
):
    seed_everything(seed)
    device = torch.device(device_num)
    cfg = TissueNetConfig(zarr_dir)
    marker_emb = cfg.load_marker_embeddings_array(svd_path=svd_embeddings_path)
    n_cls = cfg.NUM_CELLTYPES

    # Cache-local FOV-grouped loaders (sampler off = natural distribution).
    train_loader, val_loader, _ = create_dataloader(
        zarr_dir=zarr_dir,
        dct_config=cfg,
        batch_size=batch_size,
        num_dropout_channels=0,
        num_workers=num_workers,
        split_file=split_file,
        use_weighted_sampler=False,
        fov_grouped_train=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
        pin_memory=True,
    )

    ck = torch.load(pretrained_path, map_location=device, weights_only=False)  # trusted local ckpt (pretrain.py writes numpy scalars)
    cc = ck.get("config", {}) if isinstance(ck, dict) else {}
    model = create_model(
        cfg,
        marker_emb,
        d_model=cc.get("d_model", 256),
        resnet_base_channels=cc.get("resnet_channels", 48),
        spatial_pool_size=cc.get("spatial_pool_size", 1),
        n_heads=cc.get("n_heads", 8),
        use_conditioned_mp_head=cc.get("use_conditioned_mp_head", True),
        compat_marker0_zero=cc.get("compat_marker0_zero", True),
    )
    # We only need the backbone: the checkpoint's cell-type head is discarded and
    # a fresh residual-MLP head is trained below, so drop its ct_head.* params and
    # load the rest non-strictly. The only params left uninitialized must be the
    # fresh ct_head; anything else missing/unexpected signals a real mismatch.
    backbone_state = {
        k: v
        for k, v in (ck["model"] if "model" in ck else ck).items()
        if not k.startswith("ct_head.")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    assert not unexpected, f"unexpected keys loading backbone: {unexpected}"
    assert all(k.startswith("ct_head.") for k in missing), (
        f"backbone load left non-head params uninitialized: "
        f"{[k for k in missing if not k.startswith('ct_head.')]}"
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Extracting frozen CLS embeddings...", flush=True)
    Xtr, ytr = _extract_cls(model, train_loader, device, "train")
    Xva, yva = _extract_cls(model, val_loader, device, "val")
    Xtr = Xtr.astype(np.float32)
    Xva = Xva.astype(np.float32)
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xn = (Xtr - mu) / sd

    # Train the residual-MLP head on the natural distribution (plain CE).
    head = ResidualMLPHead(Xtr.shape[1], width=width, depth=depth, n_out=n_cls).to(
        device
    )
    Xf = torch.tensor(Xn)
    yf = torch.tensor(ytr)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    bs = 16384
    n = len(Xf)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs)
    )
    lossf = nn.CrossEntropyLoss()
    from sklearn.metrics import f1_score

    val_labels = np.unique(yva)
    Xe = torch.tensor((Xva - mu) / sd, device=device)
    for ep in range(epochs):
        head.train()
        pm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = pm[i : i + bs]
            opt.zero_grad()
            lossf(head(Xf[idx].to(device)), yf[idx].to(device)).backward()
            opt.step()
            sched.step()
    head.eval()
    with torch.no_grad():
        pred = torch.cat(
            [head(Xe[i : i + 32768]).argmax(1).cpu() for i in range(0, len(Xe), 32768)]
        ).numpy()
    macro = f1_score(yva, pred, average="macro", labels=val_labels) * 100
    print(f"resMLP head val macro-F1 = {macro:.2f} (flat, full-coverage)", flush=True)

    # Fold standardization (mu, sd) into the head's first linear so it takes raw CLS.
    with torch.no_grad():
        W = head.inp[0].weight.data.clone()
        b = head.inp[0].bias.data.clone()
        s = torch.tensor(sd.reshape(-1), device=device)
        m = torch.tensor(mu.reshape(-1), device=device)
        head.inp[0].weight.data = W / s.unsqueeze(0)
        head.inp[0].bias.data = b - (W / s.unsqueeze(0)) @ m

    # Assemble + save the deployable resMLP model.
    model.ct_head = head.to(device)
    out_config = {
        **cc,
        "ct_head_arch": "resmlp",
        "ct_head_width": int(width),
        "ct_head_depth": int(depth),
        "stage2_epochs": int(epochs),
        "stage2_lr": float(lr),
        "stage2_seed": int(seed),
        "stage2_split_file": split_file,
        "stage2_pretrained_path": pretrained_path,
        "stage2_svd_embeddings_path": svd_embeddings_path,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": out_config,
            # Bundle the canonical channel / cell-type registry (matching
            # train.py) so inference validates the ordering this checkpoint was
            # trained with, instead of falling back to the packaged-vocab SHA
            # anchor meant for the legacy released checkpoint.
            "canonical_channels": list(cfg.marker2idx.keys()),
            "ct2idx": dict(cfg.ct2idx),
        },
        output,
    )
    print(f"Saved deployable resMLP checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()

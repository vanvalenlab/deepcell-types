"""Generate OpenAI embeddings for the v8 archive channel+cell-type master lists.

Loads 283 channel names and 51 cell-type names directly from the v8 zarr root,
calls text-embedding-3-large, writes raw + SVD-reduced npz files.
"""
import os
from pathlib import Path

import click
import numpy as np
import zarr
from sklearn.decomposition import TruncatedSVD


def make_marker_prompt(name: str) -> str:
    return (
        f"The protein biomarker {name} is used in multiplexed immunofluorescence "
        f"imaging for cell type identification in tissue samples."
    )


def make_celltype_prompt(name: str) -> str:
    return (
        f"The cell type {name} as identified in multiplexed immunofluorescence "
        f"imaging of human tissue samples."
    )


@click.command()
@click.option("--zarr_path", type=str, required=True)
@click.option("--raw_output", type=str, default="embeddings/openai_3large_v8.npz")
@click.option("--svd_output", type=str, default="embeddings/svd_512_v8.npz")
@click.option("--n_components", type=int, default=512)
@click.option("--model", type=str, default="text-embedding-3-large")
def main(zarr_path, raw_output, svd_output, n_components, model):
    for line in open(".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

    from openai import OpenAI
    client = OpenAI()

    z = zarr.open_group(zarr_path, mode="r")
    marker_names = list(z.attrs["all_standardized_channels"])
    ct_names = list(z.attrs["cell_type_mapping"].keys())
    print(f"Markers: {len(marker_names)} (e.g., {marker_names[:5]})")
    print(f"Cell types: {len(ct_names)} (e.g., {ct_names[:5]})")

    prompts = [make_marker_prompt(n) for n in marker_names] + [make_celltype_prompt(n) for n in ct_names]
    print(f"Total prompts: {len(prompts)} → calling {model}...")
    resp = client.embeddings.create(input=prompts, model=model)
    all_emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
    print(f"Embedding dim: {all_emb.shape[1]}")

    n_m = len(marker_names)
    marker_emb = all_emb[:n_m]
    ct_emb = all_emb[n_m:]

    Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(raw_output, marker_embeddings=marker_emb, ct_embeddings=ct_emb,
             marker_names=np.array(marker_names), ct_names=np.array(ct_names), model=model)
    print(f"Raw saved → {raw_output}")

    k = min(n_components, min(all_emb.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    reduced = svd.fit_transform(all_emb).astype(np.float32)
    print(f"SVD {k}d, explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    marker2idx = {n: i for i, n in enumerate(marker_names)}
    ct2idx = {n: i for i, n in enumerate(ct_names)}
    np.savez(svd_output,
             marker_embeddings=reduced[:n_m],
             ct_embeddings=reduced[n_m:],
             svd_components=svd.components_,
             explained_variance_ratio=svd.explained_variance_ratio_,
             marker2idx=marker2idx,
             ct2idx=ct2idx)
    print(f"SVD saved → {svd_output}")


if __name__ == "__main__":
    main()

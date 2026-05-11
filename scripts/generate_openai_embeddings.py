"""Generate marker and cell type embeddings using OpenAI text-embedding-3-large.

Embeds biomarker/protein names with biological context prompts for better
semantic quality. Generates both raw embeddings and SVD-reduced versions.

Usage:
    export OPENAI_API_KEY=sk-...
    DATA_DIR=/data/xwang3/tissuenet-caitlin-labels.zarr \
        python -m scripts.generate_openai_embeddings \
        --output_path embeddings/openai_3large.npz \
        --svd_output_path embeddings/svd_512_v3.npz
"""

import os
import click
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

from deepcell_types.training.config import TissueNetConfig


def make_marker_prompt(name: str) -> str:
    """Create a biology-aware prompt for a marker/protein name."""
    return (
        f"The protein biomarker {name} is used in multiplexed immunofluorescence "
        f"imaging for cell type identification in tissue samples."
    )


def make_celltype_prompt(name: str) -> str:
    """Create a biology-aware prompt for a cell type name."""
    return (
        f"The cell type {name} as identified in multiplexed immunofluorescence "
        f"imaging of human tissue samples."
    )


def get_embeddings_batch(client, texts: list, model: str = "text-embedding-3-large") -> list:
    """Get embeddings for a batch of texts. OpenAI supports up to 2048 inputs."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


@click.command()
@click.option("--output_path", type=str, default="embeddings/openai_3large.npz",
              help="Path to save raw embeddings")
@click.option("--svd_output_path", type=str, default="embeddings/svd_512_v3.npz",
              help="Path to save SVD-reduced embeddings")
@click.option("--n_components", type=int, default=512,
              help="SVD components (capped by min(n_samples, n_features)-1)")
@click.option("--model", type=str, default="text-embedding-3-large",
              help="OpenAI embedding model name")
def main(output_path, svd_output_path, n_components, model):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    output_path = Path(output_path)
    svd_output_path = Path(svd_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svd_output_path.parent.mkdir(parents=True, exist_ok=True)

    dct_config = TissueNetConfig()

    # Get ordered marker names (aligned with marker2idx)
    n_markers = len(dct_config.marker2idx)
    marker_names = [""] * n_markers
    for name, idx in dct_config.marker2idx.items():
        marker_names[idx] = name
    print(f"Markers: {n_markers} (e.g., {marker_names[:5]})")

    # Get ordered cell type names (aligned with ct2idx)
    n_celltypes = len(dct_config.ct2idx)
    ct_names = [""] * n_celltypes
    for name, idx in dct_config.ct2idx.items():
        ct_names[idx] = name
    print(f"Cell types: {n_celltypes} (e.g., {ct_names[:5]})")

    # Create prompts
    marker_prompts = [make_marker_prompt(name) for name in marker_names]
    ct_prompts = [make_celltype_prompt(name) for name in ct_names]
    all_prompts = marker_prompts + ct_prompts
    print(f"Total prompts: {len(all_prompts)}")

    # Get embeddings (single batch, well under 2048 limit)
    print(f"Calling OpenAI {model}...")
    all_embeddings = get_embeddings_batch(client, all_prompts, model=model)
    embed_dim = len(all_embeddings[0])
    print(f"Embedding dim: {embed_dim}")

    marker_embeddings = np.array(all_embeddings[:n_markers], dtype=np.float32)
    ct_embeddings = np.array(all_embeddings[n_markers:], dtype=np.float32)

    # Save raw embeddings — atomic so a SIGTERM mid-write doesn't leave a
    # partial npz that fails the next training run with cryptic BadZipFile.
    from deepcell_types.training.utils import _atomic_np_savez

    _atomic_np_savez(
        Path(output_path),
        marker_embeddings=marker_embeddings,
        ct_embeddings=ct_embeddings,
        marker_names=marker_names,
        ct_names=ct_names,
        model=model,
    )
    print(f"Raw embeddings saved to {output_path}")
    print(f"  marker_embeddings: {marker_embeddings.shape}")
    print(f"  ct_embeddings: {ct_embeddings.shape}")

    # SVD reduction
    all_emb = np.concatenate([marker_embeddings, ct_embeddings], axis=0)
    actual_components = min(n_components, min(all_emb.shape) - 1)
    print(f"\nRunning TruncatedSVD with {actual_components} components...")
    svd = TruncatedSVD(n_components=actual_components, random_state=42)
    all_reduced = svd.fit_transform(all_emb)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_var:.4f}")

    marker_reduced = all_reduced[:n_markers].astype(np.float32)
    ct_reduced = all_reduced[n_markers:].astype(np.float32)

    _atomic_np_savez(
        Path(svd_output_path),
        marker_embeddings=marker_reduced,
        ct_embeddings=ct_reduced,
        svd_components=svd.components_,
        explained_variance_ratio=svd.explained_variance_ratio_,
        marker2idx={k: v for k, v in dct_config.marker2idx.items()},
        ct2idx={k: v for k, v in dct_config.ct2idx.items()},
    )
    print(f"SVD-reduced embeddings saved to {svd_output_path}")
    print(f"  marker_embeddings: {marker_reduced.shape}")
    print(f"  ct_embeddings: {ct_reduced.shape}")


if __name__ == "__main__":
    main()

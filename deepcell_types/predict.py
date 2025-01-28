import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import yaml

import torch
from torch.utils.data import DataLoader

from dct_kit.config import DCTConfig

from model import CellTypeCLIPModel
from dataset import PatchDataset

dct_config = DCTConfig()

@dataclass
class BatchData:
    sample: torch.Tensor
    ch_idx: torch.Tensor
    mask: torch.Tensor
    ct_idx: torch.Tensor
    cell_index: torch.Tensor
    fov_name: str


class PredLogger:
    def __init__(self):
        self.labels = []
        self.probs = []
        self.cell_index = []
        self.fov_name = []

    def log(self, labels, probs, cell_index, fov_name):
        self.labels.append(labels)
        self.probs.append(probs)
        self.cell_index.append(cell_index)
        self.fov_name.append(fov_name)

    def save(self, path_name):
        idx2ct = {v: k for k, v in dct_config.ct2idx.items()}
        columns = sorted(dct_config.ct2idx, key=dct_config.ct2idx.get)
        labels = np.concatenate(self.labels)
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        fov_name = np.concatenate(self.fov_name)
        df = pd.DataFrame(probs, columns=columns)
        df["cell_type_actual"] = labels
        df["cell_type_actual_str"] = df['cell_type_actual'].map(idx2ct)
        df["cell_type_pred_str"] = df[columns].idxmax(axis=1)
        df["cell_index"] = cell_index
        df["fov_name"] = fov_name
        df.drop(columns=['cell_type_actual'], inplace=True) # drop the numerical label
        df.to_csv(path_name, index=False)



def forward_one_batch(
    batch_data: BatchData,
    device: torch.device,
    model: torch.nn.Module,
    predlogger: PredLogger = None,
):
    # Move tensors to device except for fov_name
    # batch_data = BatchData(**{k: v.to(device) for k, v in batch_data.__dict__.items() if k != "fov_name"})
    batch_data = BatchData(
        sample=batch_data.sample.to(device),
        ch_idx=batch_data.ch_idx.to(device),
        mask=batch_data.mask.to(device),
        ct_idx=batch_data.ct_idx.to(device),
        cell_index=batch_data.cell_index.to(device),
        fov_name=batch_data.fov_name,
    )

    # Forward pass
    _, _, _, marker_pos_attn, probs, image_embedding = model(
        batch_data.sample,
        batch_data.ch_idx,
        batch_data.mask,
        batch_data.ct_idx,
    )

    if predlogger is not None:
        predlogger.log(
            labels=batch_data.ct_idx.detach().cpu().numpy(),
            probs=probs.cpu().detach().numpy(),
            cell_index=batch_data.cell_index.detach().cpu().numpy(),
            fov_name=batch_data.fov_name,
        )


@click.command()
@click.option("--model_name", type=str, default="model_combined_ct")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--patch_data_name", type=str, default="")
def main(model_name, device_num, patch_data_name):
    device = torch.device(device_num)
    data_dir = Path("data")
    patch_data_path = data_dir / patch_data_name

    # Load ct2embedding
    ct2embedding_dict = dct_config.get_celltype_embedding()
    
    ct_embeddings = np.zeros((len(dct_config.ct2idx), 1024), dtype=np.float32)
    for ct, ebd in ct2embedding_dict.items():
        if ct not in dct_config.ct2idx:
            continue
        idx = dct_config.ct2idx[ct]
        ct_embeddings[idx] = ebd


    # Load marker2embedding
    marker2embedding = dct_config.get_channel_embedding(
        embedding_model_name="text-embedding-3-large-1024"
    )

    marker_embeddings = np.empty_like(list(marker2embedding.values()), dtype=np.float32)
    for marker, ebd in marker2embedding.items():
        if marker not in dct_config.marker2idx:
            continue
        idx = dct_config.marker2idx[marker]
        marker_embeddings[idx] = ebd

    # if celltype_mapping.yaml exists, load it
    celltype_mapping = None
    celltype_mapping_path = data_dir / "celltype_mapping.yaml"
    if celltype_mapping_path.exists():
        with open(celltype_mapping_path, "r") as f:
            celltype_mapping = yaml.safe_load(f)

    # if channel_mapping.yaml exists, load it
    channel_mapping = None
    channel_mapping_path = data_dir / "channel_mapping.yaml"
    if channel_mapping_path.exists():
        with open(channel_mapping_path, "r") as f:
            channel_mapping = yaml.safe_load(f)

    # Load datasets
    test_dataset = PatchDataset(
        patch_data_path,
        dct_config=dct_config,
        celltype_mapping=celltype_mapping,
        channel_mapping=channel_mapping,
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=24)

    model = CellTypeCLIPModel(
        n_filters=256,
        n_heads=4,
        n_celltypes=28,
        n_domains=6,
        marker_embeddings=marker_embeddings,
        embedding_dim=1024,
        ct_embeddings=ct_embeddings,
        img_feature_extractor="conv"
    )
    model.load_state_dict(torch.load(f"model/{model_name}.pt", map_location=device))
    model.to(device)

    
    # Run prediction on test set
    model.eval()
    with torch.no_grad():
        predlogger = PredLogger()
        for batch in tqdm(test_loader, desc=f"(test)"):
            batch_data = BatchData(*batch)
            forward_one_batch(
                batch_data, device, model, predlogger=predlogger
            )

        predlogger.save(Path(f"output/{model_name}_{patch_data_path.stem}.csv"))
    
    return


if __name__ == "__main__":
    main()

import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .dct_kit.config import DCTConfig

from .model import CellTypeCLIPModel
from .dataset import PatchDataset

dct_config = DCTConfig()


class PredLogger:
    def __init__(self):
        self.probs = []
        self.cell_index = []

    def log(self, probs, cell_index):
        self.probs.append(probs)
        self.cell_index.append(cell_index)

    def get_result(self):
        idx2ct = {v: k for k, v in dct_config.ct2idx.items()}
        probs = np.concatenate(self.probs)
        cell_index = np.concatenate(self.cell_index)
        
        top_probs = np.max(probs, axis=1)
        cell_type_int_pred = np.argmax(probs, axis=1)
        cell_type_str_pred = [idx2ct[i] for i in cell_type_int_pred]
        
        return cell_type_str_pred, top_probs, cell_index


def predict(raw, mask, channel_names, mpp, model_name, device_num, batch_size=256, num_workers=24): 
    device = torch.device(device_num)

    embedding_model_name = "deepseek-r1-70b-llama-distill-q4_K_M"
    embedding_dim = 8192

    # Load ct2embedding
    ct2embedding_dict = dct_config.get_celltype_embedding(
        embedding_model_name=embedding_model_name
    )

    # ct_embeddings = np.zeros_like(list(ct2embedding_dict.values()), dtype=np.float32)
    ct_embeddings = np.zeros((len(dct_config.ct2idx), embedding_dim), dtype=np.float32)
    for ct, ebd in ct2embedding_dict.items():
        if ct not in dct_config.ct2idx:
            continue
        idx = dct_config.ct2idx[ct]
        ct_embeddings[idx] = ebd

    # Load marker2embedding
    marker2embedding = dct_config.get_channel_embedding(
        embedding_model_name=embedding_model_name
    )

    marker_embeddings = np.zeros_like(list(marker2embedding.values()), dtype=np.float32)
    # marker_embeddings = np.zeros((len(dct_config.marker2idx), embedding_dim), dtype=np.float32)
    for marker, ebd in marker2embedding.items():
        if marker not in dct_config.marker2idx:
            print("bad_marker?", marker)
        idx = dct_config.marker2idx[marker]
        marker_embeddings[idx] = ebd
    
    # Load model
    model = CellTypeCLIPModel(
        n_filters=256,
        n_heads=4,
        n_celltypes=len(dct_config.ct2idx),
        n_domains=9,
        marker_embeddings=marker_embeddings,
        embedding_dim=embedding_dim,
        ct_embeddings=ct_embeddings,
        img_feature_extractor="conv"
    )

    model_path = str(Path.home() / ".deepcell" / "models" / f"{model_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    pred_logger = PredLogger()

    # Initialize dataset
    dataset = PatchDataset(raw, mask, channel_names, mpp, dct_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Run prediction on test set
    model.eval()
    with torch.no_grad():
        for sample, ch_idx, attn_mask, cell_index in tqdm(data_loader, desc=f"(inference)"):
            _, _, _, _, probs, _ = model(
                sample.to(device),
                ch_idx.to(device),
                attn_mask.to(device),
            )

            pred_logger.log(
                probs=probs.cpu().detach().numpy(),
                cell_index=cell_index.detach().cpu().numpy(),
            )


        result = pred_logger.get_result()
        cell_types = result[0]
    
    return cell_types



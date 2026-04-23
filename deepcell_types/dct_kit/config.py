import os
from pathlib import Path
import yaml
import json

from .utils import flatten_nested_dict


class DCTConfig:
    CHANNEL_ALIASES = {
        "CgA": "CHGA",
        "DC-SIGN": "CD209",
        "DCSIGN": "CD209",
        "Galectin-9": "Galectin9",
        "HO-1": "HO1",
        "Pan-Cytokeratin": "PanCK",
        "PANCK": "PanCK",
    }

    def __init__(self, profile="legacy"):
        if profile not in {"legacy", "canonical"}:
            raise ValueError("profile must be 'legacy' or 'canonical'")
        self.profile = profile
        self.SEED = 0
        self.MAX_NUM_CHANNELS = 80 if profile == "canonical" else 75
        self.BATCH_SIZE = 400
        self.MAX_CHUNK_PER_CT_PER_DATASET = 25
        self.PERCENTILE_THRESHOLD = 99.0

        self.HIST_NORM_KERNEL_SIZE = 128
        self.CROP_SIZE = 32 if profile == "canonical" else 64
        self.OUTPUT_SIZE = self.CROP_SIZE

        self.STANDARD_MPP_RESOLUTION = 0.5

        self.data_folder = Path(os.path.dirname(__file__)) / "config"
        self._ct2idx, self._core_celltypes = self._load_ct2idx_and_core_celltypes()

        self._master_channels = self._load_master_channels()

        if profile == "canonical":
            self._marker2idx = {ch: idx for idx, ch in enumerate(self.master_channels)}
        else:
            embedding_model_name = "deepseek-r1-70b-llama-distill-q4_K_M"
            marker2embedding = self.get_channel_embedding(
                embedding_model_name=embedding_model_name
            )
            self._marker2idx = {ch: idx for idx, ch in enumerate(marker2embedding)}
        # self._dataset2idx = {k: idx for idx, k in enumerate(self.celltype_mapping.keys())}
        self.NUM_CELLTYPES = len(self.ct2idx)
        self.NUM_DOMAINS = 8 if profile == "canonical" else 9

        # Default channel mapping containing all recognized marker name aliases to the
        # names recognized by the model
        with open(self.data_folder / "channel_mapping.yaml") as fh:
            channel_mapping = yaml.safe_load(fh) or {}
        if profile == "canonical":
            canonical_mapping = {ch: ch for ch in self.master_channels}
            canonical_mapping.update(
                {
                    alias: target
                    for alias, target in self.CHANNEL_ALIASES.items()
                    if target in self._marker2idx
                }
            )
            canonical_mapping.update(
                {
                    alias: target
                    for alias, target in channel_mapping.items()
                    if target in self._marker2idx
                }
            )
            channel_mapping = canonical_mapping
        self.channel_mapping = channel_mapping

    @property
    def ct2idx(self):
        return self._ct2idx

    @property
    def domain2idx(self):
        return self._domain2idx

    @property
    def marker2idx(self):
        return self._marker2idx

    @property
    def dataset2idx(self):
        return self._dataset2idx

    @property
    def core_celltypes(self):
        return self._core_celltypes

    def _load_ct2idx_and_core_celltypes(self):
        filename = (
            "canonical_celltypes.yaml"
            if self.profile == "canonical"
            else "core_celltypes.yaml"
        )
        with open(self.data_folder / filename, "r") as f:
            core_celltypes = yaml.safe_load(f)

        if isinstance(core_celltypes, list):
            return {ct: idx for idx, ct in enumerate(core_celltypes)}, core_celltypes

        master_celltype_list = flatten_nested_dict(core_celltypes)
        master_celltype_list_updated = []
        for celltype in master_celltype_list:
            if celltype != "Cell":
                master_celltype_list_updated.append(celltype)

        ct2idx = {ct: idx for idx, ct in enumerate(master_celltype_list_updated)}

        return ct2idx, core_celltypes

    @property
    def master_channels(self):
        return self._master_channels

    def _load_master_channels(self):
        filename = (
            "canonical_channels.yaml"
            if self.profile == "canonical"
            else "master_channels.yaml"
        )
        with open(self.data_folder / filename, "r") as f:
            master_channels = yaml.load(f, Loader=yaml.FullLoader)
        return master_channels

    def get_channel_embedding(self, embedding_model_name="text-embedding-3-large-1024"):
        """Get the channel embedding from the json file."""
        with open(
            self.data_folder / f"marker_embeddings-{embedding_model_name}.json", "r"
        ) as f:
            channel_embedding = json.load(f)
        return channel_embedding

    def get_celltype_embedding(
        self, embedding_model_name="text-embedding-3-large-1024"
    ):
        """Get the celltype embedding from the json file."""
        with open(
            self.data_folder / f"celltype_embeddings-{embedding_model_name}.json", "r"
        ) as f:
            ct2embedding_dict = json.load(f)
        return ct2embedding_dict

    def get_tct_mapping(self):
        """Get the tissue to celltype mapping from the yaml file."""
        with open(self.data_folder / "tissue_celltype_mapping_merged.yaml", "r") as f:
            tct = yaml.safe_load(f)
        return tct


if __name__ == "__main__":
    dct_config = DCTConfig()

    print(dct_config.__dict__)

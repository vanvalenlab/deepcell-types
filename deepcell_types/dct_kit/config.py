import os
from pathlib import Path
import yaml
import json

from .utils import flatten_nested_dict


class DCTConfig:
    def __init__(self):
        self.SEED = 0
        self.MAX_NUM_CHANNELS = 75
        self.BATCH_SIZE = 400
        self.MAX_CHUNK_PER_CT_PER_DATASET = 25
        self.PERCENTILE_THRESHOLD = 99.0

        self.HIST_NORM_KERNEL_SIZE = 128
        self.CROP_SIZE = 64

        self.STANDARD_MPP_RESOLUTION = 0.5

        self.data_folder = Path(os.path.dirname(__file__)) / "config"
        self._ct2idx, self._core_celltypes = self._load_ct2idx_and_core_celltypes()

        self._master_channels = self._load_master_channels()

        embedding_model_name = "deepseek-r1-70b-llama-distill-q4_K_M"
        marker2embedding = self.get_channel_embedding(
            embedding_model_name=embedding_model_name
        )
        self._domain2idx = {domain:idx for idx, domain in enumerate(sorted(set(list(self.domain_mapping.values()))))}
        # self._marker2idx = {ch: idx for idx, ch in enumerate(self.master_channels)}
        self._marker2idx = {ch: idx for idx, ch in enumerate(marker2embedding)}
        # self._dataset2idx = {k: idx for idx, k in enumerate(self.celltype_mapping.keys())}
        self.NUM_CELLTYPES = len(self.ct2idx)
        self.NUM_DOMAINS = len(self.domain2idx)
        

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
        with open(self.data_folder / "core_celltypes.yaml", "r") as f:
            core_celltypes = yaml.safe_load(f)

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
        with open(self.data_folder / "master_channels.yaml", "r") as f:
            master_channels = yaml.load(f, Loader=yaml.FullLoader)
        return master_channels

    
    def get_channel_embedding(self, embedding_model_name="text-embedding-3-large-1024"):
        """Get the channel embedding from the json file.
        """
        with open(self.data_folder / f"marker_embeddings-{embedding_model_name}.json", "r") as f:
            channel_embedding = json.load(f)
        return channel_embedding
    
    def get_celltype_embedding(self, embedding_model_name="text-embedding-3-large-1024"):
        """Get the celltype embedding from the json file.
        """
        with open(self.data_folder / f"celltype_embeddings-{embedding_model_name}.json", "r") as f:
            ct2embedding_dict = json.load(f)
        return ct2embedding_dict
    

if __name__ == "__main__":
    dct_config = DCTConfig()

    print(dct_config.__dict__)

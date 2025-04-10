import numpy as np
import zarr
from pprint import pprint
from packaging import version

# This version of the archive was created with zarr v2, so can only be
# ready with zarr v2
assert version.parse(zarr.__version__).major < 3

# Open archive
zname = "/data/hubmap.zarr.zip"
store = zarr.storage.ZipStore(zname, mode="r", allowZip64=True)
z = zarr.open_group(store=store, mode="r")

# Choose a dataset
ds = z["HBM685_PCCJ_427"]
print(ds.tree())

# Dataset attributes
pprint(dict(ds.attrs))

# Load data
chnames = ds["image"].attrs["channels"]  # Channel names
mpp = ds["image"].attrs['mpp'] # Micron per pixel
mask = ds["segmentations/torch_mesmer"]  # Segmentation mask (W, H)
img = ds["image"]  # Multiplexed image, (C, W, H)

# Convert both to float32
mask = mask.astype(np.float32)
img = img.astype(np.float32)

print(mask.shape, mask.dtype)
print(img.shape, img.dtype)

# Choose model and device
model_name = "deepcell-types_specific_ct_v0.1"
device_num = "cuda:0"

# Run prediction
from deepcell_types import predict
cell_types = predict(img, mask, chnames, mpp, model_name, device_num)

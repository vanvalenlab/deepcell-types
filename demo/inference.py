from tqdm import tqdm
from collections import defaultdict
import numpy as np
import zarr
from pprint import pprint
from packaging import version

# Open archive
zname = "/data2/hubmap.zarr"
z = zarr.open(zname, mode="r")

# Choose a dataset
k = "HBM224_RCHM_744"
ds = z[k]
print(ds.tree())

# Dataset attributes
pprint(dict(ds.attrs))

# Load data
chnames = ds["image"].attrs["channels"]  # Channel names
mpp = ds["image"].attrs['mpp'] # Micron per pixel
mask = ds["segmentations/torch_mesmer"][:]  # Segmentation mask (W, H)
img = ds["image"][:]  # Multiplexed image, (C, W, H)

# Convert both to float32
mask = mask.astype(np.float32)
img = img.astype(np.float32)

print(mask.shape, mask.dtype)
print(img.shape, img.dtype)

# Choose model and device
model_name = "model_c_patch2_entire_dataset"
device_num = "cuda:0"

# Run prediction
from deepcell_types import predict
cell_types = predict(img, mask, chnames, mpp, model_name, device_num, num_workers=1)

# Visualize
#import napari
#cl = [(mn, mx) for mn, mx in zip(img.min(axis=(1, 2)), img.max(axis=(1, 2)))]
#nim = napari.view_image(img, channel_axis=0, name=chnames, contrast_limits=cl);
#seglyr = nim.add_labels(mask.astype(np.uint32));
#seglyr.contour = 1
#
## Visualize celltypes
#uniq_ct = np.unique(cell_types)
#cmask = mask > 0
#idx_to_pred = dict(enumerate(cell_types, start=1))
#
#labels_by_celltype = defaultdict(list)
#for idx, ct in idx_to_pred.items():
#    labels_by_celltype[ct].append(idx)
#
#ctmasks = {}
#for i, (cell_type, cell_indices) in tqdm(enumerate(labels_by_celltype.items())):
#    lbl = i + 1
#    # Compute binary mask for given celltype
#    ctmask = np.zeros_like(mask, dtype=np.uint8)
#    idx_set = set(cell_indices)
#    ctmask[cmask] = [lbl if pix in idx_set else 0 for pix in mask[cmask]]
#    # Store results
#    ctmasks[cell_type] = ctmask
#
#for ctname, ctmask in ctmasks.items():
#    nim.add_labels(ctmask, name=f"{ctname.upper()} | {len(labels_by_celltype[ctname])}")

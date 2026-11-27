# ==============================================================================
# File    : dataset.py
# Purpose : PyTorch dataset for patches and masks
#
# Author  : Sophia Mengjia Li
# Created : 2026-11-27
# ==============================================================================

import torch

import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclasses
from collections.abc import Callable
from torch.utils.data import Dataset
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)  # 100MB


# =====| Class Definitions |===================================================

@dataclass(frozen = True)
class DataDirectories:
    """
    Locations of the processed (normalized) patches and masks, laid out as 
    `<dir>/<sample_id>/<file>`. Built once by the CLI and passed explicitly.
    """

    patches     : Path
    vessel_masks: Path
    tissue_masks: Path


class VesselPatchDataset(Dataset):
    """
    Patch dataset for vessel segmentation in bone marrow IHC biopsies.

    Loads Vahadane-normalized RGB patches with binary vessel and tissue masks. 
    Transforms injected externally per train/validation split.

    Tissue masks applied in loss computation (not as model input) to restrict 
    gradient updates to valid tissue regions, preventing background bias in 
    severely class-imbalanced segmentation tasks.
    """

    def __init__(self,
                 manifest: pd.DataFrame,
                 dirs: DataDirectories,
                 transform: Callable | None = None) -> None:
        super().__init__()

        self.records         = manifest.reset_index(drop = True)
        self.patch_dir       = dirs.patches
        self.vessel_mask_dir = dirs.vessel_masks
        self.tissue_mask_dir = dirs.tissue_masks
        self.transform       = transform

    
    def __getitem__(self, idx: int) -> dict:

        # Load the normalized RGB patch from the patch directory
        row        = self.records.iloc[idx]
        patch_path = self.patch_dir / row['sample_id'] / row['patch_name']
        with Image.open(patch_path) as img: patch = np.array(img.convert('RGB'))

        # Load the binary masks (0/255 uint8, mode 'L')
        vessel_path = (self.vessel_mask_dir / row['sample_id'] 
                                            / row['vessel_mask'])
        with Image.open(vessel_path) as img: vessel_mask = np.array(img)

        tissue_path = (self.tissue_mask_dir / row['sample_id'] 
                                            / row['tissue_mask'])
        with Image.open(tissue_path) as img: tissue_mask = np.array(img)

        # Apply joint spatial transforms to patch and vessel masks (not tissue)
        if self.transform is not None:
            transformed = self.transform(image = patch, mask = vessel_mask)
            patch, vessel_mask = transformed['image'], transformed['mask']

        # Convert patch and masks to tensors: HWC -> CHW, normalize [0, 1]
        patch       = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        vessel_mask = torch.from_numpy(vessel_mask).unsqueeze(0).float() / 255.0
        tissue_mask = torch.from_numpy(tissue_mask).squeeze(0).float() / 255.0

        return {
            "patch"      : patch,
            "vessel_mask": vessel_mask,
            "tissue_mask": tissue_mask,
            "has_vessel" : bool(row["has_vessel"]),
            "patch_id"   : str(row["patch_id"]),
            "sample_id"  : str(row["sample_id"]),
            "patch_name" : str(row["patch_name"])
        }
    

    def __len__(self) -> int:
        return len(self.records)

# [END]
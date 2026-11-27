# ==============================================================================
# File    : datamodule.py
# Purpose : PyTorch Lightning DataModule for one cross-validated fold
# Author  : Sophia Mengjia Li
# Created : 2026-11-27
# ==============================================================================

import logging

import lightning.pytorch as pl
import pandas as pd

from torch.utils.data import DataLoader

from fibroscore.data.sampler import PositiveWeightedSampler
from fibroscore.data.dataset import DataDirectories, VesselPatchDataset
from fibroscore.data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# =====| Primary Entrypoint |===================================================

class VesselSegmentationDdataModule(pl.LightningDataModule):
    """
    Patient-level k-fold DataModule for vessel segmentation.

    Training split uses PositiveWeightedSampler with per-patient 
    inverse-frequency weighting to prevent high-density patient dominance.

    Validation split uses identity transforms and sequential sampling to 
    produce unbiased performance estimates.
    """

    def __init__(self,
                 manifest   : pd.DataFrame,
                 dirs       : DataDirectories,
                 fold       : int,
                 batch_size : int,
                 num_workers: int,
                 pos_prob   : float = 0.5,
                 seed       : int = 42) -> None:
        super().__init__()

        self.manifest   = manifest.copy()
        self.dirs       = dirs
        self.fold       = fold
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.pos_prob    = pos_prob
        self.seed        = seed
 
        self.train_dataset: VesselPatchDataset | None = None
        self.val_dataset  : VesselPatchDataset | None = None


    def setup(self, stage: str | None = None) -> None:
        """
        Partition manifest into train and validation splits by fold index.
        """

        # Explicitly cast folds to integers to avoid silent failures
        self.manifest['fold'] = self.manifest['fold'].astype(int)

        # Extract datasets based on the current fold
        train_manifest = self.manifest[self.manifest['fold'] != self.fold]
        val_manifeset  = self.manifest[self.manifest['fold'] == self.fold]

        train_manifest = train_manifest.reset_index(drop = True)
        val_manifest   = val_manifest.reset_index(drop = True)

        # Initialize the datasets with transform protocols
        self.train_dataset = VesselPatchDataset(
            manifest  = train_manifest,
            dirs      = self.dirs,
            transform = get_train_transforms()
        )
        self.val_dataset = VesselPatchDataset(
            manifest  = val_manifest,
            dirs      = self.dirs,
            transform = get_val_transforms()
        )

        return

    # ~~~~~| DataLoaders |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def train_dataloader(self) -> DataLoader:
        """
        Training DataLoader with PositiveWeightedSampler.
        """

        assert self.train_dataset is not None, ("Must call setup()")

        sampler = PositiveWeightedSampler(
            manifest = self.train_dataset.records,
            pos_prob = self.pos_prob,
            seed     = self.seed
        )

        return DataLoader(
            self.train_dataset,
            batch_size         = self.batch_size,
            sampler            = sampler,
            num_workers        = self.num_workers,
            shuffle            = False,
            pin_memory         = self.num_workers > 0,
            persistent_workers = self.num_workers > 0,
            drop_last          = True
        )


    def val_dataloader(self) -> DataLoader:
        """
        Validation DataLoader with sequential sampling.
        """

        assert self.val_dataset is not None, ("Must call setup()")

        return DataLoader(
            self.val_dataset,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = self.num_workers > 0,
            persistent_workers = self.num_workers > 0,
            drop_last          = False
        )
      

# [END]
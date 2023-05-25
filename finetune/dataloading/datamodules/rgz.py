import sys
import torchvision.transforms as T
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
from mae.dataloading.datamodules.fits import FITS_DataModule
from typing import Dict, Union
import torch

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader

from paths import Path_Handler

from astroaugmentations.datasets.MiraBest_F import (
    MBFRFull,
    MBFRConfident,
    MiraBest_F,
    MBFRUncertain,
    MiraBest_FITS,
)


# TODO get aug parameters like center_crop from loaded model


class FineTuning_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # override default paths via config if desired
        paths = Path_Handler(**config.get("paths_to_override", {}))
        path_dict = paths._dict()
        self.path = path_dict[config["dataset"]]

        self.config = config

        self.mu, self.sig = config["data"]["mu"], config["data"]["sig"]

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.config["finetune"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            prefetch_factor=self.config["data"]["prefetch_factor"],
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"],
            batch_size=200,
            num_workers=8,
            prefetch_factor=30,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loaders = [
            DataLoader(data, **self.config["val_dataloader"]) for data in self.data["test"].values()
        ]
        return loaders


class RGZ_DataModule_Finetune(FineTuning_DataModule):
    def __init__(self, config):
        super().__init__(config)

        # Cropping
        center_crop = config["augmentations"]["center_crop_size"]

        self.T_train = T.Compose(
            [
                T.RandomRotation(180),
                T.CenterCrop(center_crop),
                T.RandomResizedCrop(center_crop, scale=(0.9, 1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

        self.T_test = T.Compose(
            [
                T.CenterCrop(center_crop),
                T.ToTensor(),
                T.Normalize(self.mu, self.sig),
            ]
        )

    def prepare_data(self):
        MiraBest_F(self.path, train=None, download=True)

    def setup(self, stage=None):
        # Get test set which is held out and does not change
        # self.data["test"] = MBFRConfident(
        #     self.path,
        #     aug_type="torchvision",
        #     train=False,
        #     # test_size=self.config["finetune"]["test_size"],
        #     test_size=None,
        #     transform=self.T_test,
        # )

        self.data["test"] = OrderedDict(
            {
                "MB_conf_test": MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                    transform=self.T_test,
                ),
                "MB_unc_test": MBFRUncertain(
                    self.path,
                    aug_type="torchvision",
                    train=False,
                    test_size=None,
                    transform=self.T_test,
                ),
            },
        )

        if self.config["finetune"]["val_size"] != 0:
            data = MBFRConfident(self.path, aug_type="torchvision", train=True)
            idx = np.arange(len(data))
            idx_train, idx_val = train_test_split(
                idx,
                test_size=self.config["finetune"]["val_size"],
                stratify=data.full_targets,
                random_state=self.config["finetune"]["seed"],
            )

            self.data["train"] = Subset(
                MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                    transform=self.T_train,
                ),
                idx_train,
            )

            self.data["val"] = Subset(
                MBFRConfident(
                    self.path,
                    aug_type="torchvision",
                    train=True,
                    test_size=None,
                    transform=self.T_test,
                ),
                idx_val,
            )

        else:
            self.data["train"] = MBFRConfident(
                self.path, aug_type="torchvision", train=True, transform=self.T_train
            )
            self.data["val"] = MBFRConfident(
                self.path, aug_type="torchvision", train=True, transform=self.T_test
            )


def confident_only(df):
    df = df.loc[df["class"].isin(["FR1", "FR2"])]
    df = df.loc[df["confidence"] == "confident"]
    return df.reset_index(drop=True)


def no_hybrid(df):
    df = df.loc[df["class"].isin(["FR1", "FR2"])]
    return df.reset_index(drop=True)


class MiraBest_FITS_DataModule_Finetune(FITS_DataModule, FineTuning_DataModule):
    def __init__(
        self,
        config,
        path,
        batch_size: int,
        num_workers: int = 1,
        prefetch_factor: int = 8,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        img_size: bool = 128,
        MiraBest_FITS_root: str = "/share/nas2_5/mbowles/_data/MiraBest_FITS",
        data_type: Union[str, type] = torch.float32,
        astroaugment: bool = True,
        fft: bool = True,  # TODO
        png: bool = False,  # TODO
        nchan: int = 3,
        **kwargs,
    ):
        super(MiraBest_FITS_DataModule_Finetune, self).__init__(
            config=config,
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            img_size=img_size,
            MiraBest_FITS_root=MiraBest_FITS_root,
            data_type=data_type,
            astroaugment=astroaugment,
            fft=fft,
            png=png,
            nchan=nchan,
            **kwargs,
        )

    def setup(self):
        self.data["test"] = OrderedDict(
            {
                "MB_conf_test": MiraBest_FITS(
                    root=self.MiraBest_FITS_root,
                    train=False,
                    transform=self.eval_transform,
                    data_type=self.data_type,
                    df_filter=confident_only,
                ),
                "MB_nohybrid_test": MiraBest_FITS(
                    root=self.MiraBest_FITS_root,
                    train=False,
                    transform=self.eval_transform,
                    data_type=self.data_type,
                    df_filter=no_hybrid,
                ),
            }
        )
        self.data["train"] = MiraBest_FITS(
            self.path,
            train=True,
            test_size=self.config["finetune"]["val_size"],
            transform=self.train_transform,
            data_type=self.data_type,
            df_filter=confident_only,
        )

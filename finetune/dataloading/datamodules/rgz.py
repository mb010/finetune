import sys
import torchvision.transforms as T
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
from mae.dataloading.datamodules.vision import Base_DataModule
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

import albumentations as A
import astroaugmentations as AA
from astroaugmentations.datasets.MiraBest_F import (
    MBFRFull,
    MBFRConfident,
    MiraBest_F,
    MBFRUncertain,
    MiraBest_FITS,
)


class FineTuning_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int = 8,
        prefetch_factor: int = 30,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs,
    ):
        """
        Args:
            path: path to dataset
            batch_size: batch size
        """
        super().__init__()

        self.path = path
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        self.data = {}

    def prepare_data(self):
        return

    def train_dataloader(self):
        loader = DataLoader(
            self.data["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loaders = [
            DataLoader(
                data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=False,
            )
            for data in self.data["test"].values()
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


class MiraBest_FITS_DataModule_Finetune(FineTuning_DataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int = 1,
        prefetch_factor: int = 8,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        img_size: bool = 128,
        data_type: Union[str, type] = torch.float32,
        astroaugment: bool = True,
        fft: bool = True,  # TODO
        png: bool = False,  # TODO
        nchan: int = 3,
        test_size: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            path,
            batch_size,
            num_workers,
            prefetch_factor,
            persistent_workers,
            pin_memory,
            **kwargs,
        )
        self.mu = (0.485, 0.456, 0.406)
        self.sig = (0.229, 0.224, 0.225)
        self.img_size = img_size
        self.data_type = {
            "torch.float32": torch.float32,
            "16-mixed": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            "32-true": torch.float32,
            "64-true": torch.float64,
            64: torch.float64,
            32: torch.float32,
            16: torch.float16,
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.float16,
            "bf16": torch.bfloat16,
        }[data_type]
        self.astroaugment = astroaugment
        self.fft = fft
        self.png = png
        self.nchan = nchan
        self.test_size = test_size
        self.train_transform, self.test_transform, self.eval_transform = self._build_transforms()

    def _repeat_array(self, arr, repetitions):
        arr = arr[np.newaxis, :]
        return np.repeat(arr, repetitions, axis=0)

    def _build_transforms(self):
        # Handle fft and channel shape conditions
        if self.fft:
            if self.nchan == 3:
                out = [np.real, np.imag, np.angle]
            elif self.nchan == 2:
                out = [np.real, np.imag]
        else:
            out = [np.asarray for i in range(self.nchan)]
        # Handle astroaugment and fft parameters
        train_transform = [A.CenterCrop(self.img_size, self.img_size)]
        test_transform = [A.CenterCrop(self.img_size, self.img_size)]
        eval_transform = [A.CenterCrop(self.img_size, self.img_size)]
        if self.astroaugment:
            train_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(
                        dropout_p=0.8,
                        dropout_mag=0.5,  # RFI Overflagging
                        noise_p=0.5,
                        noise_mag=0.5,  # Noise Injection
                        rfi_p=0.5,
                        rfi_mag=1,
                        rfi_prob=0.01,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
            test_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
            eval_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
        else:
            train_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
            test_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
            eval_transform.append(
                A.Lambda(
                    name="UVAugmentation",
                    image=AA.image_domain.radio.UVAugmentation(  # Fourrier transform the same way as before.
                        dropout_p=0.0,  # RFI Overflagging
                        noise_p=0.0,  # Noise Injection
                        rfi_p=0.0,  # RFI injection
                        fft=self.fft,
                        out=out,
                    ),
                    p=1,
                )
            )
        # Handle png parameter
        if self.png:
            train_transform.append(A.Lambda(name="png_norm", image=AA.image_domain.NaivePNGnorm(), p=1))
            test_transform.append(A.Lambda(name="png_norm", image=AA.image_domain.NaivePNGnorm(), p=1))
            eval_transform.append(A.Lambda(name="png_norm", image=AA.image_domain.NaivePNGnorm(), p=1))

        return A.Compose(train_transform), A.Compose(test_transform), A.Compose(eval_transform)

    def setup(self, stage=None):
        self.data["test"] = OrderedDict(
            {
                "MB_conf_test": MiraBest_FITS(
                    root=self.path,
                    train=False,
                    test_size=self.test_size,
                    transform=self.eval_transform,
                    data_type=self.data_type,
                    df_filter=confident_only,
                    aug_type="albumentations",
                ),
                "MB_nohybrid_test": MiraBest_FITS(
                    root=self.path,
                    train=False,
                    test_size=self.test_size,
                    transform=self.eval_transform,
                    data_type=self.data_type,
                    df_filter=no_hybrid,
                    aug_type="albumentations",
                ),
            }
        )
        self.data["train"] = MiraBest_FITS(
            self.path,
            train=True,
            test_size=self.test_size,
            transform=self.train_transform,
            data_type=self.data_type,
            df_filter=confident_only,
            aug_type="albumentations",
        )

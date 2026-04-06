import os
import random
import torch

from enum import StrEnum
from torchvision.io import decode_image, ImageReadMode
import torchvision.transforms.v2.functional as F


class VariantDirectory(StrEnum):
    LR_DIFFICULT = "lr_difficult"
    LR_MILD = "lr_mild"
    LR_WILD = "lr_wild"
    LR_X8 = "lr_x8"


class VariantSuffix(StrEnum):
    LR_DIFFICULT = "x4d"
    LR_MILD = "x4m"
    LR_WILD = "x4w"
    LR_X8 = "x8"


class Div2k2018TrainDataset(torch.utils.data.Dataset):

    PATH_PREFIX = "data/train/div2k_2018"
    HR_DATA_SIZE = 800
    VARIANTS = [
        (VariantDirectory.LR_DIFFICULT, VariantSuffix.LR_DIFFICULT),
        (VariantDirectory.LR_MILD, VariantSuffix.LR_MILD),
        (VariantDirectory.LR_WILD, VariantSuffix.LR_WILD),
        (VariantDirectory.LR_X8, VariantSuffix.LR_X8),
    ]

    def __init__(self, scale: int, patch_size: int):
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.VARIANTS) * self.HR_DATA_SIZE

    def __getitem__(self, index):
        variant_index = int(index / self.HR_DATA_SIZE)
        index_in_variant = (index % self.HR_DATA_SIZE) + 1
        variant_directory, variant_suffix = self.VARIANTS[variant_index]

        # lr image [C, W, H]
        lr_name = f"{index_in_variant:04d}{variant_suffix}{random.randint(1, 4) if variant_directory is VariantDirectory.LR_WILD else ""}.png"
        lr_path = os.path.join(self.PATH_PREFIX, variant_directory, lr_name)
        lr_image = decode_image(lr_path, mode=ImageReadMode.RGB).float() / 255.0

        # hr image
        hr_name = f"{index_in_variant:04d}.png"
        hr_path = os.path.join(self.PATH_PREFIX, "hr", hr_name)
        hr_image = decode_image(hr_path, mode=ImageReadMode.RGB).float() / 255.0

        # Random crop
        _, h_lr, w_lr = lr_image.shape
        top = random.randint(0, h_lr - self.patch_size)
        left = random.randint(0, w_lr - self.patch_size)

        lr_patch = F.crop(lr_image, top, left, self.patch_size, self.patch_size)
        hr_patch = F.crop(
            hr_image, top * self.scale, left * self.scale, self.patch_size * self.scale, self.patch_size * self.scale
        )

        return lr_patch, hr_patch

import os
import random
import torch

from enum import StrEnum
from torchvision.io import decode_image, ImageReadMode
import torchvision.transforms.v2.functional as F


class VariantDirectory(StrEnum):
    LR_NN_2X = "lr_nn_2x"
    LR_NN_4X = "lr_nn_4x"
    LR_NN_8X = "lr_nn_8x"
    LR_BILIN_2X = "lr_bilin_2x"
    LR_BILIN_4X = "lr_bilin_4x"
    LR_BILIN_8X = "lr_bilin_8x"
    LR_BICUB_2X = "lr_bicub_2x"
    LR_BICUB_4X = "lr_bicub_4x"
    LR_BICUB_8X = "lr_bicub_8x"
    LR_LAN_2X = "lr_lan_2x"
    LR_LAN_4X = "lr_lan_4x"
    LR_LAN_8X = "lr_lan_8x"
    LR_DIFFICULT = "lr_difficult"
    LR_MILD = "lr_mild"
    LR_WILD = "lr_wild"
    LR_X8 = "lr_x8"


class VariantSuffix(StrEnum):
    LR_DIFFICULT = "x4d"
    LR_MILD = "x4m"
    LR_WILD = "x4w"
    LR_X8 = "x8"
    NONE = ""


class Div2k2018Dataset(torch.utils.data.Dataset):

    VARIANTS = {
        2: [
            (VariantDirectory.LR_NN_2X, VariantSuffix.NONE),
            (VariantDirectory.LR_BILIN_2X, VariantSuffix.NONE),
            (VariantDirectory.LR_BICUB_2X, VariantSuffix.NONE),
            (VariantDirectory.LR_LAN_2X, VariantSuffix.NONE),
        ],
        4: [
            (VariantDirectory.LR_NN_4X, VariantSuffix.NONE),
            (VariantDirectory.LR_BILIN_4X, VariantSuffix.NONE),
            (VariantDirectory.LR_BICUB_4X, VariantSuffix.NONE),
            (VariantDirectory.LR_LAN_4X, VariantSuffix.NONE),
        ],
        8: [
            (VariantDirectory.LR_NN_8X, VariantSuffix.NONE),
            (VariantDirectory.LR_BILIN_8X, VariantSuffix.NONE),
            (VariantDirectory.LR_BICUB_8X, VariantSuffix.NONE),
            (VariantDirectory.LR_LAN_8X, VariantSuffix.NONE),
        ],
    }

    def __init__(self, scale: int, patch_size: int | None = None, scale_variant: str | int = "all"):
        self.scale = scale
        self.patch_size = patch_size

        if scale_variant == "all":
            self.active_variants = [v for sublist in self.VARIANTS.values() for v in sublist]
            self.is_multi_scale = True
        else:
            self.active_variants = self.VARIANTS.get(scale_variant, [])
            self.is_multi_scale = False
            self.fixed_scale = int(scale_variant)

        if not self.active_variants:
            raise ValueError(f"Scale {scale_variant} is not supported by Div2K2018 dataset")

    def __len__(self):
        return len(self.active_variants) * self.HR_DATA_SIZE

    def __getitem__(self, index):
        variant_index = index // self.HR_DATA_SIZE
        index_in_variant = (index % self.HR_DATA_SIZE) + 1
        variant_directory, variant_suffix = self.active_variants[variant_index]

        # lr image [C, W, H]
        lr_name = f"{index_in_variant:04d}{variant_suffix}{random.randint(1, 4) if variant_directory is VariantDirectory.LR_WILD else ""}.png"
        lr_path = os.path.join(self.PATH_PREFIX, variant_directory, lr_name)
        lr_image = decode_image(lr_path, mode=ImageReadMode.RGB).float() / 255.0

        # hr image
        hr_name = f"{index_in_variant:04d}.png"
        hr_path = os.path.join(self.PATH_PREFIX, "hr", hr_name)
        hr_image = decode_image(hr_path, mode=ImageReadMode.RGB).float() / 255.0

        if not self.patch_size:
            return lr_image, hr_image

        # Random crop
        _, h_lr, w_lr = lr_image.shape
        current_patch_h = min(h_lr, self.patch_size)
        current_patch_w = min(w_lr, self.patch_size)
        top = random.randint(0, h_lr - current_patch_h)
        left = random.randint(0, w_lr - current_patch_w)

        lr_patch = F.crop(lr_image, top, left, current_patch_h, current_patch_w)
        hr_patch = F.crop(
            hr_image, top * self.scale, left * self.scale, current_patch_h * self.scale, current_patch_w * self.scale
        )

        return lr_patch, hr_patch


class Div2k2018TestDataset(Div2k2018Dataset):

    PATH_PREFIX = "data/test/div2k_2018"
    HR_DATA_SIZE = 100

    def __init__(self, scale: int, patch_size: int | None = None, scale_variant: str | int = "all"):
        super().__init__(scale=scale, patch_size=patch_size, scale_variant=scale_variant)


class Div2k2018TrainDataset(Div2k2018Dataset):

    PATH_PREFIX = "data/train/div2k_2018"
    HR_DATA_SIZE = 800

    def __init__(self, scale: int, patch_size: int | None = None, scale_variant: str | int = "all"):
        super().__init__(scale=scale, patch_size=patch_size, scale_variant=scale_variant)

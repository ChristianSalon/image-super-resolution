import glob
import numpy as np
import os
import torch

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.v2.functional as F


class RedsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, seq_length=15, patch_size=64, scale=4, is_train=True):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

        self.video_list = sorted(glob.glob(os.path.join(root_dir, "*")))

        self.frames_per_video = 100
        self.num_segments = self.frames_per_video - self.seq_length + 1

    def __len__(self):
        return len(self.video_list) * self.num_segments

    def __getitem__(self, idx):
        video_idx = idx // self.num_segments
        start_frame = idx % self.num_segments

        video_path = self.video_list[video_idx]

        hr_seq = []

        # Load sequence of frames
        for i in range(start_frame, start_frame + self.seq_length):
            img_path = os.path.join(video_path, f"{i:08d}.png")
            hr_img = decode_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
            hr_seq.append(hr_img)

        # Random crop
        _, w, h = hr_seq[0].shape
        if self.is_train:
            # Random top left corner for the HR patch
            th, tw = self.patch_size * self.scale, self.patch_size * self.scale
            i = np.random.randint(0, h - th + 1)
            j = np.random.randint(0, w - tw + 1)
        else:
            # Center crop for validation
            i, j = (h - self.patch_size * self.scale) // 2, (w - self.patch_size * self.scale) // 2

        # Apply crops and create LR versions
        hr_tensor_seq = []
        lr_tensor_seq = []

        for hr_img in hr_seq:
            # Crop HR
            hr_patch = F.crop(hr_img, i, j, self.patch_size * self.scale, self.patch_size * self.scale)

            # Create LR by bicubic downsampling
            lr_patch = F.resize(hr_patch, [self.patch_size, self.patch_size], interpolation=InterpolationMode.BICUBIC)

            hr_tensor_seq.append(hr_patch)
            lr_tensor_seq.append(lr_patch)

        # Final shapes: [T, C, H, W]
        return torch.stack(lr_tensor_seq), torch.stack(hr_tensor_seq)

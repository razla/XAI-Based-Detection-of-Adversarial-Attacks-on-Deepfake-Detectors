"""

Author: Andreas Rössler
"""
import os
import random

import numpy as np
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from PIL import Image
import torch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        transforms.functional.normalize(tensor, self.mean, self.std, inplace=True)
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub(m).div(s)
        #     # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_transformer(face_policy: str, patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
            A.Resize(height=patch_size, width=patch_size, always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    if train:
        aug_transformations = [
            A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
                ]),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
                A.ImageCompression(quality_lower=50, quality_upper=99),
            ], )
        ]
    else:
        aug_transformations = []

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf


EfficientNetB4ST_default_data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'to_tensor': transforms.Compose([
        transforms.ToTensor()
    ]),
    'normalize': transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #     "to_tensor": A.Compose([
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ToTensorV2()
    #     ]),
    #     "un_normalize": A.Compose([
    #         A.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    #         A.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    #     ])
}

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}


class ImageXaiFolder(Dataset):
    def __init__(self, original_path=None, original_xai_path=None, attacked_path=None, attacked_xai_path=None,
                 transform=None,
                 black_xai=False, black_img=False):
        super(ImageXaiFolder, self).__init__()
        self.original_path = []
        self.original_xai_path = []
        self.attacked_path = []
        self.attacked_xai_path = []

        self.black_xai = black_xai
        self.black_img = black_img

        if original_path is not None or original_xai_path is not None:
            original_paths = os.listdir(original_path)
            original_xai_paths = os.listdir(original_xai_path)
            self.original_path = original_path
            self.original_xai_path = original_xai_path
            self.original_images = ['original-' + image for image in original_paths if image.endswith(".jpg")]
            self.original_xai = [image for image in original_xai_paths if image.endswith(".jpg")]
        if attacked_path is not None or attacked_xai_path is not None:
            attacked_paths = os.listdir(attacked_path)
            attacked_xai_paths = os.listdir(attacked_xai_path)
            self.attacked_path = attacked_path
            self.attacked_xai_path = attacked_xai_path
            self.attacked_images = ['attacked-' + image for image in attacked_paths if image.endswith(".jpg")]
            self.attacked_xai = [image for image in attacked_xai_paths if image.endswith(".jpg")]

        self.images = self.original_images + self.attacked_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if image_path.find('original-') != -1:
            base_name = image_path.split('-')[1]
            image_path = os.path.join(self.original_path, base_name)
            xai_path = os.path.join(self.original_xai_path, base_name)
            label = np.array([1.0, 0.0])
        elif image_path.find('attacked-') != -1:
            base_name = image_path.split('-')[1]
            image_path = os.path.join(self.attacked_path, base_name)
            xai_path = os.path.join(self.attacked_xai_path, base_name)
            label = np.array([0.0, 1.0])

        image = self.loader(image_path)
        xai_map = self.loader(xai_path)

        if self.transform is not None:
            image = self.transform(image)
            xai_map = self.transform(xai_map)
            if self.black_xai:
                xai_map = torch.zeros_like(xai_map)
            if self.black_img:
                image = torch.zeros_like(image)

        return image, xai_map, label

    def loader(self, path):
        return Image.open(path)

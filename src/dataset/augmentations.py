# File: src/dataset/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_aug(img_size: int):
    
    return A.Compose([
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.10,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.60
        ),
        A.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4
        ),
        A.CoarseDropout(
            max_holes=1,
            max_height=img_size // 6,
            max_width=img_size // 6,
            min_holes=1,
            min_height=img_size // 10,
            min_width=img_size // 10,
            p=0.3
        ),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_aug(img_size: int):
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])

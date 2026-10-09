"""Аугментации, имитирующие съёмку на обычный смартфон в амбулатории.

Освещение и баланс белого меняются (ColorJitter), фото бывают смазаны
(GaussianBlur) и сильно сжаты мессенджером (RandomJPEG). Геометрия —
повороты и отражения: у образования на коже нет «верха» и «низа».
"""

from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomJPEG:
    """Пересжимает изображение в JPEG со случайным качеством."""

    def __init__(self, quality=(30, 90), p: float = 0.3):
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self) -> str:
        return f"RandomJPEG(quality={self.quality}, p={self.p})"


def _to_tensor():
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_train_transform(size: int = 224, strength: str = "phone"):
    if strength == "none":
        return build_eval_transform(size)
    ops = [
        v2.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomApply([v2.RandomRotation(30)], p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04),
    ]
    if strength == "phone":
        ops += [
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
            RandomJPEG(quality=(30, 90), p=0.3),
        ]
    return v2.Compose(ops + _to_tensor())


def build_eval_transform(size: int = 224):
    return v2.Compose([v2.Resize((size, size), antialias=True)] + _to_tensor())

"""PyTorch Dataset для снимков PAD-UFES-20 (+ вектор анкеты)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path: str, max_side: int | None = 768) -> Image.Image:
    """Открывает снимок и сразу уменьшает большие фото — ускоряет обучение."""
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return img


class ImageTabDataset(Dataset):
    def __init__(self, paths, labels, tab: np.ndarray | None = None, transform=None,
                 cache_images: bool = False):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tab = None if tab is None else np.asarray(tab, dtype=np.float32)
        self.transform = transform
        self._cache: dict[int, Image.Image] | None = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.paths)

    def load_image(self, i: int) -> Image.Image:
        if self._cache is not None:
            if i not in self._cache:
                self._cache[i] = load_rgb(self.paths[i])
            return self._cache[i].copy()
        return load_rgb(self.paths[i])

    def __getitem__(self, i: int) -> dict:
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        item = {"image": img, "label": self.labels[i], "idx": i}
        item["tab"] = torch.from_numpy(self.tab[i]) if self.tab is not None else torch.zeros(0)
        return item

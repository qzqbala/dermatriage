"""ISIC 2024 / SLICE-3D: 400 000+ фрагментов 3D-фото всего тела смартфонного качества.

Источник: Kurtansky et al., Scientific Data 11, 884 (2024);
Kaggle: https://www.kaggle.com/competitions/isic-2024-challenge

Нужны два файла:
    data/isic2024/train-metadata.csv
    data/isic2024/train-image.hdf5   (ключ — isic_id, значение — байты JPEG)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler

ISIC_USECOLS = [
    "isic_id", "target", "patient_id", "age_approx", "sex",
    "anatom_site_general", "clin_size_long_diam_mm",
]


def load_isic_metadata(root: str | Path, verbose: bool = True) -> pd.DataFrame:
    root = Path(root)
    csv = root / "train-metadata.csv"
    h5 = root / "train-image.hdf5"
    for p in (csv, h5):
        if not p.exists():
            raise FileNotFoundError(
                f"Не найден {p}. Скачайте данные ISIC 2024 (см. README, раздел «Данные»)."
            )
    header = pd.read_csv(csv, nrows=0).columns
    usecols = [c for c in ISIC_USECOLS if c in header]
    df = pd.read_csv(csv, usecols=usecols, low_memory=False)
    df["label"] = df["target"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    if verbose:
        print(f"[ISIC] {len(df)} снимков, {df['label'].sum()} злокачественных, "
              f"{df['patient_id'].nunique()} пациентов")
    return df


class ISICHDF5Dataset(Dataset):
    """Читает JPEG из HDF5. Файл открывается лениво — отдельно в каждом worker'е."""

    def __init__(self, h5_path: str | Path, isic_ids, labels, transform=None):
        self.h5_path = str(h5_path)
        self.ids = list(isic_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.transform = transform
        self._h5 = None

    def __len__(self) -> int:
        return len(self.ids)

    def _file(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def load_image(self, i: int) -> Image.Image:
        raw = self._file()[self.ids[i]][()]
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")

    def __getitem__(self, i: int) -> dict:
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": self.labels[i], "idx": i}

    def __getstate__(self):
        # h5py-объект нельзя передать в процесс-worker; откроется заново.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


class BalancedNegativeSampler(Sampler):
    """Каждую эпоху берёт все положительные примеры и свежую случайную выборку отрицательных.

    При доле злокачественных ~0,1% это даёт сбалансированные эпохи и за всё
    обучение модель видит большую часть отрицательных примеров.
    """

    def __init__(self, labels, neg_per_pos: int = 20, seed: int = 0):
        labels = np.asarray(labels)
        self.pos = np.flatnonzero(labels == 1)
        self.neg = np.flatnonzero(labels == 0)
        self.n_neg = min(len(self.neg), max(1, len(self.pos)) * neg_per_pos)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pos) + self.n_neg

    def __iter__(self):
        neg = self.rng.choice(self.neg, size=self.n_neg, replace=False)
        idx = np.concatenate([self.pos, neg])
        self.rng.shuffle(idx)
        return iter(idx.tolist())

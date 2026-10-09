"""Кодировщик анкеты пациента в числовой вектор.

Важно для честной оценки: статистики (медиана, среднее, словари категорий)
считаются ТОЛЬКО на обучающей выборке (fit), а затем применяются к val/test
и в демо (transform). Кодировщик сохраняется в JSON вместе с моделью.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .pad_ufes import normalize_ternary

DEFAULT_NUMERIC = ["age"]
DEFAULT_CATEGORICAL = ["gender", "region"]
DEFAULT_TERNARY = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]

UNK = "UNK"


@dataclass
class TabularEncoder:
    numeric: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC))
    categorical: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORICAL))
    ternary: list[str] = field(default_factory=lambda: list(DEFAULT_TERNARY))
    min_count: int = 5
    stats: dict[str, Any] = field(default_factory=dict)

    # ---------- обучение ----------
    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        stats: dict[str, Any] = {"numeric": {}, "categorical": {}}
        for col in self.numeric:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)
            med = float(vals.median()) if vals.notna().any() else 0.0
            filled = vals.fillna(med)
            std = float(filled.std()) if len(filled) > 1 else 1.0
            stats["numeric"][col] = {"median": med, "mean": float(filled.mean()) if len(filled) else 0.0,
                                     "std": std if std > 1e-6 else 1.0}
        for col in self.categorical:
            vals = df[col].fillna(UNK).astype(str).str.upper() if col in df else pd.Series(dtype=str)
            counts = vals.value_counts()
            vocab = sorted(v for v, c in counts.items() if c >= self.min_count and v != UNK)
            stats["categorical"][col] = vocab
        self.stats = stats
        return self

    # ---------- применение ----------
    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for col in self.numeric:
            names += [f"{col}", f"{col}__missing"]
        for col in self.categorical:
            names += [f"{col}={v}" for v in self.stats["categorical"][col]] + [f"{col}=other/unk"]
        for col in self.ternary:
            names += [f"{col}=yes", f"{col}=no"]
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.stats:
            raise RuntimeError("Сначала вызовите fit() на обучающей выборке")
        n = len(df)
        parts = []
        for col in self.numeric:
            st = self.stats["numeric"][col]
            vals = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series([np.nan] * n)
            miss = vals.isna().to_numpy().astype(np.float32)
            z = ((vals.fillna(st["median"]) - st["mean"]) / st["std"]).to_numpy().astype(np.float32)
            parts += [z[:, None], miss[:, None]]
        for col in self.categorical:
            vocab = self.stats["categorical"][col]
            vals = df[col].fillna(UNK).astype(str).str.upper().to_numpy() if col in df else np.array([UNK] * n)
            onehot = np.zeros((n, len(vocab) + 1), dtype=np.float32)
            pos = {v: i for i, v in enumerate(vocab)}
            for i, v in enumerate(vals):
                onehot[i, pos.get(v, len(vocab))] = 1.0
            parts.append(onehot)
        for col in self.ternary:
            vals = df[col].map(normalize_ternary).to_numpy() if col in df else np.array(["unk"] * n)
            parts.append(np.stack([(vals == "yes"), (vals == "no")], axis=1).astype(np.float32))
        return np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float32)

    def transform_one(self, answers: dict[str, Any]) -> np.ndarray:
        """Кодирует одну анкету (словарь поле → значение), например из веб-демо."""
        return self.transform(pd.DataFrame([answers]))[0]

    # ---------- сохранение ----------
    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical, "ternary": self.ternary,
                "min_count": self.min_count, "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "TabularEncoder":
        return cls(numeric=d["numeric"], categorical=d["categorical"], ternary=d["ternary"],
                   min_count=d.get("min_count", 5), stats=d["stats"])

"""Разбиение на train / val / test строго по пациентам.

У одного пациента бывает несколько образований и снимков. Если снимки одного
пациента попадут и в обучение, и в тест, модель «узнает» пациента, и метрики
будут завышены (утечка данных). Поэтому делим по patient_id и проверяем это.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 20  # 20 частей по 5%: 3 части — test, 3 — val, 14 — train (70/15/15)


def patient_split(
    df: pd.DataFrame,
    seed: int,
    group_col: str = "patient_id",
    label_col: str = "label",
    val_folds: int = 3,
    test_folds: int = 3,
) -> pd.Series:
    """Возвращает Series со значениями 'train' / 'val' / 'test' (индекс как у df).

    Стратификация по метке сохраняет долю злокачественных примерно одинаковой
    во всех трёх выборках; разные seed дают разные разбиения.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df[label_col], groups=df[group_col])):
        fold[idx] = k
    rng = np.random.default_rng(seed)
    order = rng.permutation(N_FOLDS)
    test_ids = set(order[:test_folds])
    val_ids = set(order[test_folds:test_folds + val_folds])
    split = np.where(np.isin(fold, list(test_ids)), "test",
                     np.where(np.isin(fold, list(val_ids)), "val", "train"))
    out = pd.Series(split, index=df.index, name="split")
    check_no_patient_leakage(df, out, group_col)
    return out


def check_no_patient_leakage(df: pd.DataFrame, split: pd.Series, group_col: str = "patient_id") -> None:
    groups = {s: set(df.loc[split == s, group_col]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise AssertionError(f"Утечка: {len(overlap)} пациентов одновременно в {a} и {b}")


def split_summary(df: pd.DataFrame, split: pd.Series, group_col: str = "patient_id") -> pd.DataFrame:
    rows = []
    for s in ("train", "val", "test"):
        part = df[split == s]
        rows.append({
            "split": s,
            "images": len(part),
            "patients": part[group_col].nunique(),
            "malignant": int(part["label"].sum()),
            "malignant_share": float(part["label"].mean()) if len(part) else float("nan"),
        })
    return pd.DataFrame(rows)

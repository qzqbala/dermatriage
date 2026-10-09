"""PAD-UFES-20: фото со смартфона + клинические данные пациента.

Источник: Pacheco et al., Data in Brief 32 (2020) 106221,
https://data.mendeley.com/datasets/zr7vgbcyr2/1

Ожидаемая структура после распаковки (вложенность папок может отличаться —
изображения ищутся рекурсивно по имени файла из столбца img_id):

    data/pad_ufes_20/
        metadata.csv
        images/imgs_part_1/*.png, imgs_part_2/*.png, imgs_part_3/*.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAD_DIAGNOSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
PAD_DIAGNOSIS_NAMES_RU = {
    "ACK": "актинический кератоз",
    "BCC": "базальноклеточный рак",
    "MEL": "меланома",
    "NEV": "невус (родинка)",
    "SCC": "плоскоклеточный рак",
    "SEK": "себорейный кератоз",
}
# Злокачественные по умолчанию. ACK — предраковое состояние: в отчёте
# отдельно показывается, какую долю ACK система направляет к специалисту.
MALIGNANT_DEFAULT = ["BCC", "MEL", "SCC"]

# Поля анкеты «да / нет / неизвестно» — их фельдшер может спросить у пациента.
SYMPTOM_FIELDS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
REQUIRED_COLUMNS = ["patient_id", "img_id", "diagnostic"]

_YES = {"true", "1", "1.0", "yes", "y", "да"}
_NO = {"false", "0", "0.0", "no", "n", "нет"}


def normalize_ternary(value) -> str:
    """Приводит значение к 'yes' / 'no' / 'unk'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unk"
    s = str(value).strip().lower()
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    return "unk"


def find_metadata_csv(root: Path) -> Path:
    direct = root / "metadata.csv"
    if direct.exists():
        return direct
    found = sorted(root.rglob("metadata.csv"))
    if not found:
        raise FileNotFoundError(
            f"Не найден metadata.csv в {root}. Скачайте PAD-UFES-20 и распакуйте в эту папку "
            "(см. README, раздел «Данные»)."
        )
    return found[0]


def index_images(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG"):
        for p in root.rglob(ext):
            index.setdefault(p.name, p)
    return index


def load_pad_metadata(
    root: str | Path,
    positive_classes: list[str] | None = None,
    drop_missing_images: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Читает metadata.csv, нормализует поля, находит пути к снимкам, добавляет метку.

    Возвращаемые служебные столбцы:
        image_path — путь к файлу снимка;
        label      — 1, если диагноз из positive_classes;
        fitzpatrick — фототип (float, NaN если неизвестен).
    """
    root = Path(root)
    positive_classes = [c.upper() for c in (positive_classes or MALIGNANT_DEFAULT)]
    df = pd.read_csv(find_metadata_csv(root))
    df.columns = [c.strip().lower() for c in df.columns]
    # В оригинальном файле столбец называется 'fitspatrick' (с опечаткой).
    if "fitspatrick" in df.columns and "fitzpatrick" not in df.columns:
        df = df.rename(columns={"fitspatrick": "fitzpatrick"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В metadata.csv нет обязательных столбцов: {missing}. Столбцы: {list(df.columns)}")

    df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()
    unknown = sorted(set(df["diagnostic"]) - set(PAD_DIAGNOSES))
    if unknown and verbose:
        print(f"[PAD] Внимание: неизвестные диагнозы {unknown}")

    for col in SYMPTOM_FIELDS:
        if col in df.columns:
            df[col] = df[col].map(normalize_ternary)
        else:
            df[col] = "unk"
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("UNK").astype(str).str.strip().str.upper()
    if "region" in df.columns:
        df["region"] = df["region"].fillna("UNK").astype(str).str.strip().str.upper()
    for col in ("age", "fitzpatrick", "diameter_1", "diameter_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fitzpatrick" not in df.columns:
        df["fitzpatrick"] = np.nan

    index = index_images(root)
    df["image_path"] = df["img_id"].astype(str).map(lambda n: str(index[n]) if n in index else None)
    n_missing = int(df["image_path"].isna().sum())
    if n_missing and verbose:
        print(f"[PAD] Не найдено файлов для {n_missing} из {len(df)} записей.")
    if drop_missing_images:
        df = df[df["image_path"].notna()].reset_index(drop=True)

    df["label"] = df["diagnostic"].isin(positive_classes).astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def red_flags(df: pd.DataFrame, fields=("bleed", "changed", "grew")) -> np.ndarray:
    """Клиническое правило «красных флагов»: кровоточит, менялось или растёт.

    Используется как нечисловая базовая линия и как правило безопасности поверх модели.
    """
    mask = np.zeros(len(df), dtype=bool)
    for f in fields:
        if f in df.columns:
            mask |= (df[f] == "yes").to_numpy()
    return mask

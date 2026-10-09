"""Шаг 0. Проверяет, что данные скачаны и разложены правильно.

    python scripts/check_data.py --pad data/pad_ufes_20 --isic data/isic2024
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from dermatriage.data.pad_ufes import PAD_DIAGNOSES, load_pad_metadata
from dermatriage.data.splits import patient_split, split_summary
from dermatriage.pipeline import resolve


def check_pad(root) -> bool:
    print(f"\n=== PAD-UFES-20: {root}")
    try:
        df = load_pad_metadata(root, drop_missing_images=False)
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА: {e}")
        return False
    n_img = int(df["image_path"].notna().sum())
    print(f"Записей: {len(df)}, найдено снимков: {n_img}, пациентов: {df['patient_id'].nunique()}")
    print("Диагнозы:", df["diagnostic"].value_counts().reindex(PAD_DIAGNOSES).fillna(0).astype(int).to_dict())
    print(f"Доля злокачественных (BCC+MEL+SCC): {df['label'].mean():.1%}")
    print(f"Фототип (Фицпатрик) известен для {df['fitzpatrick'].notna().sum()} записей")
    if n_img < len(df):
        print(f"ВНИМАНИЕ: для {len(df) - n_img} записей нет файла снимка. Проверьте, что распакованы "
              "и внешний архив, и вложенные архивы images/imgs_part_*.zip.")
    df = df[df["image_path"].notna()].reset_index(drop=True)
    print("\nРазбиение по пациентам (seed 0):")
    print(split_summary(df, patient_split(df, seed=0)).to_string(index=False))
    return n_img > 0


def check_isic(root) -> bool:
    print(f"\n=== ISIC 2024 (SLICE-3D): {root}")
    try:
        import h5py

        from dermatriage.data.isic2024 import ISICHDF5Dataset, load_isic_metadata

        df = load_isic_metadata(root)
        ds = ISICHDF5Dataset(root / "train-image.hdf5", df["isic_id"].head(3), df["label"].head(3))
        img = ds.load_image(0)
        print(f"Тестовое чтение снимка из HDF5: OK, размер {img.size}")
        with h5py.File(root / "train-image.hdf5", "r") as f:
            print(f"Снимков в HDF5: {len(f.keys())}")
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА: {e}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default="data/pad_ufes_20")
    ap.add_argument("--isic", default="data/isic2024")
    ap.add_argument("--skip-isic", action="store_true")
    args = ap.parse_args()
    ok = check_pad(resolve(args.pad))
    if not args.skip_isic:
        ok = check_isic(resolve(args.isic)) and ok
    print("\nИтог:", "всё в порядке" if ok else "есть проблемы — см. сообщения выше")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

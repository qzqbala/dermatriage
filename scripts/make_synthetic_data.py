"""Создаёт маленькие СИНТЕТИЧЕСКИЕ наборы с той же структурой файлов, что у PAD-UFES-20 и ISIC 2024.

Нужны только для автоматической проверки, что весь пайплайн запускается
(tests/test_pipeline.py). Это не медицинские данные; результаты на них
ничего не значат и в отчёт не идут.

    python scripts/make_synthetic_data.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DIAG = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
DIAG_P = [0.32, 0.37, 0.03, 0.10, 0.08, 0.10]
REGIONS = ["FACE", "NOSE", "BACK", "CHEST", "ARM", "FOREARM", "HAND", "NECK", "EAR", "THIGH"]


def lesion_image(rng, malignant: bool, size: int) -> Image.Image:
    skin = np.array([rng.integers(150, 235), rng.integers(110, 190), rng.integers(90, 160)])
    arr = np.clip(skin + rng.normal(0, 8, (size, size, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    c = size / 2
    r = size * (0.22 if malignant else 0.15)
    if malignant:  # неровный край, темнее, с красноватым оттенком
        pts = [(c + r * (1 + 0.35 * rng.standard_normal()) * np.cos(a),
                c + r * (1 + 0.35 * rng.standard_normal()) * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 14)]
        d.polygon(pts, fill=(int(rng.integers(60, 110)), 30, 30))
    else:
        d.ellipse([c - r, c - r, c + r, c + r], fill=(int(rng.integers(110, 150)), 80, 60))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def make_pad(out: Path, n_patients: int, rng) -> None:
    rows = []
    img_dirs = [out / "images" / f"imgs_part_{k}" for k in (1, 2, 3)]
    for d in img_dirs:
        d.mkdir(parents=True, exist_ok=True)
    lesion_id = 0
    for p in range(n_patients):
        pid = f"PAT_{1000 + p}"
        age = int(rng.integers(20, 90))
        gender = rng.choice(["MALE", "FEMALE"]) if rng.random() > 0.3 else np.nan
        fitz = float(rng.choice([1, 2, 3, 4, 5, 6], p=[.15, .4, .3, .1, .03, .02])) if rng.random() > 0.3 else np.nan
        for _ in range(int(rng.integers(1, 4))):
            lesion_id += 1
            dx = rng.choice(DIAG, p=DIAG_P)
            mal = dx in ("BCC", "MEL", "SCC")
            img_id = f"{pid}_{lesion_id}_{rng.integers(100, 999)}.png"
            size = int(rng.choice([160, 224, 300]))
            lesion_image(rng, mal, size).save(img_dirs[lesion_id % 3] / img_id)

            def yn(p_yes):
                return rng.choice(["True", "False", "UNK"], p=[p_yes, 0.9 - p_yes, 0.1])

            rows.append({
                "patient_id": pid, "lesion_id": lesion_id, "smoke": np.nan, "drink": np.nan,
                "background_father": np.nan, "background_mother": np.nan, "age": age,
                "pesticide": np.nan, "gender": gender, "skin_cancer_history": np.nan,
                "cancer_history": np.nan, "has_piped_water": np.nan, "has_sewage_system": np.nan,
                "fitspatrick": fitz, "region": rng.choice(REGIONS), "diameter_1": np.nan,
                "diameter_2": np.nan, "diagnostic": dx,
                "itch": yn(0.5), "grew": yn(0.55 if mal else 0.25), "hurt": yn(0.3),
                "changed": yn(0.4 if mal else 0.15), "bleed": yn(0.45 if mal else 0.1),
                "elevation": yn(0.6), "img_id": img_id, "biopsed": bool(mal or rng.random() < 0.5),
            })
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    print(f"PAD (синтетика): {len(rows)} снимков, {n_patients} пациентов → {out}")


def make_isic(out: Path, n_images: int, rng) -> None:
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(out / "train-image.hdf5", "w") as f:
        for i in range(n_images):
            iid = f"ISIC_{9000000 + i}"
            mal = rng.random() < 0.08
            buf = io.BytesIO()
            lesion_image(rng, mal, 128).save(buf, format="JPEG", quality=90)
            f.create_dataset(iid, data=np.void(buf.getvalue()))
            rows.append({"isic_id": iid, "target": int(mal), "patient_id": f"IP_{rng.integers(0, 60):07d}",
                         "age_approx": float(rng.integers(30, 85)), "sex": rng.choice(["male", "female"]),
                         "anatom_site_general": rng.choice(["posterior torso", "lower extremity", "head/neck"]),
                         "clin_size_long_diam_mm": float(rng.uniform(2, 10))})
    pd.DataFrame(rows).to_csv(out / "train-metadata.csv", index=False)
    print(f"ISIC (синтетика): {n_images} снимков → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-patients", type=int, default=150)
    ap.add_argument("--n-isic", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    make_pad(out / "pad_ufes_20", args.n_patients, rng)
    make_isic(out / "isic2024", args.n_isic, rng)


if __name__ == "__main__":
    main()

"""Подбирает пороги проверки качества снимка по реальным фото PAD-UFES-20 (только train).

    python scripts/calibrate_quality.py --config configs/pad_base.yaml

Логика: снимки датасета считаем «приемлемыми». Порог резкости — 1-й перцентиль,
диапазон яркости — 0,5–99,5 перцентили. Тогда проверка браковала бы ~1–2% снимков
обучающего набора, а заметно худшие реальные фото — отсекала бы.
Результат: configs/quality_limits.json (подхватывается демо автоматически) и гистограммы.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image

from dermatriage.config import load_config
from dermatriage.pipeline import prepare_pad, resolve
from dermatriage.plotting import BENIGN, save, setup_style
from dermatriage.quality import DEFAULT_LIMITS, check_quality
from dermatriage.utils import save_json

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pad_base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="configs/quality_limits.json")
    ap.add_argument("--fig-dir", default="reports/quality")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    df = prepare_pad(cfg, seed=0)
    train = df[df["split"] == "train"]
    rows = []
    for p in train["image_path"]:
        with Image.open(p) as im:
            q = check_quality(im.convert("RGB"))
        rows.append({"sharpness": q["sharpness"], "brightness": q["brightness"], "overexposed": q["overexposed"]})
    t = pd.DataFrame(rows)
    limits = dict(DEFAULT_LIMITS)
    limits["min_sharpness"] = float(np.percentile(t["sharpness"], 1))
    limits["min_brightness"] = float(np.percentile(t["brightness"], 0.5))
    limits["max_brightness"] = float(np.percentile(t["brightness"], 99.5))
    limits["max_overexposed"] = float(max(np.percentile(t["overexposed"], 99.5), 0.05))
    rejected = sum(not check_quality_from_row(r, limits) for _, r in t.iterrows())
    save_json({"limits": limits, "calibrated_on": f"PAD-UFES-20 train, {len(t)} снимков",
               "train_rejected_share": rejected / len(t),
               "stats": t.describe().round(2).to_dict()}, resolve(args.out))

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(t["sharpness"], bins=50, color=BENIGN)
    axes[0].axvline(limits["min_sharpness"], color="#e34948", linewidth=1.5)
    axes[0].set_title("Резкость (дисперсия лапласиана)")
    axes[1].hist(t["brightness"], bins=50, color=BENIGN)
    for v in (limits["min_brightness"], limits["max_brightness"]):
        axes[1].axvline(v, color="#e34948", linewidth=1.5)
    axes[1].set_title("Средняя яркость")
    save(fig, resolve(args.fig_dir) / "quality_calibration.png")
    print(f"Пороги сохранены в {resolve(args.out)}; на train отбраковано {rejected / len(t):.1%} снимков")
    print(limits)


def check_quality_from_row(r, lim) -> bool:
    return (r["sharpness"] >= lim["min_sharpness"] and lim["min_brightness"] <= r["brightness"]
            <= lim["max_brightness"] and r["overexposed"] <= lim["max_overexposed"])


if __name__ == "__main__":
    main()

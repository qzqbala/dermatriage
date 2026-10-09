"""Шаг 1. Разведочный анализ данных (EDA) + проверка на «короткие пути».

    python scripts/eda.py --pad data/pad_ufes_20 --isic data/isic2024 --out reports/eda

Результат: графики reports/eda/*.png и сводка reports/eda/eda_summary.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from dermatriage.data.pad_ufes import (MALIGNANT_DEFAULT, PAD_DIAGNOSES, PAD_DIAGNOSIS_NAMES_RU,
                                       SYMPTOM_FIELDS, load_pad_metadata)
from dermatriage.pipeline import resolve
from dermatriage.plotting import BENIGN, MALIGNANT, SERIES, TEXT_2, save, setup_style
from dermatriage.utils import save_json

import matplotlib.pyplot as plt  # noqa: E402

SYMPTOM_RU = {"itch": "зуд", "grew": "растёт", "hurt": "болит", "changed": "изменялось",
              "bleed": "кровоточит", "elevation": "возвышается"}


def label_name(v: int) -> str:
    return "злокачественные" if v == 1 else "доброкачественные"


def fig_diagnoses(df, out):
    counts = df["diagnostic"].value_counts().reindex(PAD_DIAGNOSES).fillna(0).astype(int).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [MALIGNANT if d in MALIGNANT_DEFAULT else BENIGN for d in counts.index]
    ax.barh([f"{d} — {PAD_DIAGNOSIS_NAMES_RU[d]}" for d in counts.index], counts.values, color=colors, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.01, i, str(v), va="center", fontsize=9, color=TEXT_2)
    ax.set_title("PAD-UFES-20: снимков по диагнозам")
    ax.set_xlabel("снимков")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MALIGNANT, label="злокачественные (направить)"),
                       Patch(color=BENIGN, label="доброкачественные / предрак")], loc="lower right")
    return save(fig, out / "pad_diagnoses.png")


def fig_images_per_patient(df, out):
    per = df.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.arange(1, per.max() + 2) - 0.5
    ax.hist(per, bins=bins, color=BENIGN, rwidth=0.85)
    ax.set_title("Снимков на одного пациента")
    ax.set_xlabel("снимков у пациента")
    ax.set_ylabel("пациентов")
    return save(fig, out / "pad_images_per_patient.png"), per


def fig_age(df, out):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.arange(0, 101, 5)
    for lab, color in ((0, BENIGN), (1, MALIGNANT)):
        ax.hist(df.loc[df["label"] == lab, "age"].dropna(), bins=bins, histtype="step", linewidth=2,
                color=color, label=label_name(lab), density=True)
    ax.set_title("Возраст пациентов")
    ax.set_xlabel("лет")
    ax.set_ylabel("доля")
    ax.legend()
    return save(fig, out / "pad_age.png")


def fig_symptoms(df, out):
    rows = []
    for f in SYMPTOM_FIELDS:
        for lab in (0, 1):
            part = df[df["label"] == lab][f]
            known = part[part != "unk"]
            rows.append({"field": f, "label": lab,
                         "yes_share": (known == "yes").mean() if len(known) else np.nan,
                         "unk_share": (part == "unk").mean()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(SYMPTOM_FIELDS))
    w = 0.38
    for k, (lab, color) in enumerate(((0, BENIGN), (1, MALIGNANT))):
        vals = t[t["label"] == lab].set_index("field").loc[SYMPTOM_FIELDS, "yes_share"].to_numpy()
        ax.bar(x + (k - 0.5) * w, vals * 100, width=w - 0.04, color=color, label=label_name(lab))
    ax.set_xticks(x, [SYMPTOM_RU[f] for f in SYMPTOM_FIELDS])
    ax.set_ylabel("% ответивших «да»")
    ax.set_title("Анкета: доля ответов «да» (среди известных)")
    ax.legend()
    return save(fig, out / "pad_symptoms.png"), t


def fig_regions(df, out, top: int = 12):
    tab = pd.crosstab(df["region"], df["label"])
    tab = tab.reindex(columns=[0, 1], fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(tab))
    ax.barh(y, tab[0], color=BENIGN, height=0.6, label=label_name(0))
    ax.barh(y, tab[1], left=tab[0], color=MALIGNANT, height=0.6, label=label_name(1),
            edgecolor="#fcfcfb", linewidth=1.5)
    ax.set_yticks(y, tab.index)
    ax.set_xlabel("снимков")
    ax.set_title(f"Локализация образования (топ-{top})")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, out / "pad_regions.png")


def fig_fitzpatrick(df, out):
    f = df["fitzpatrick"].map(lambda v: "неизв." if pd.isna(v) else str(int(v)))
    order = [str(i) for i in range(1, 7)] + ["неизв."]
    tab = pd.crosstab(f, df["label"]).reindex(index=order, columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, tab[0], width=w - 0.04, color=BENIGN, label=label_name(0))
    ax.bar(x + w / 2, tab[1], width=w - 0.04, color=MALIGNANT, label=label_name(1))
    ax.set_xticks(x, order)
    ax.set_xlabel("фототип по Фицпатрику")
    ax.set_ylabel("снимков")
    ax.set_title("Фототип кожи: данные для анализа справедливости")
    ax.legend()
    return save(fig, out / "pad_fitzpatrick.png"), tab


def fig_missingness(df, raw_cols, out):
    miss = df.groupby("diagnostic")[raw_cols].apply(lambda g: g.isna().mean()).reindex(PAD_DIAGNOSES)
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.55 * len(raw_cols)), 3.4))
    im = ax.imshow(miss.to_numpy() * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(raw_cols)), raw_cols, rotation=60, ha="right")
    ax.set_yticks(range(len(miss)), miss.index)
    ax.grid(False)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            v = miss.iat[i, j] * 100
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="% пропусков")
    ax.set_title("Пропуски в метаданных по диагнозам")
    return save(fig, out / "pad_missingness.png"), miss


def shortcut_check(df, raw_cols) -> dict:
    """Предсказывает метку ТОЛЬКО по факту пропусков в метаданных.

    Если AUC заметно выше 0,5, значит способ сбора данных связан с диагнозом
    (например, анкету заполняли полнее для биопсированных). Такие поля опасны:
    модель может выучить «кто заполнял анкету», а не признаки болезни.
    """
    X = df[raw_cols].isna().astype(float).to_numpy()
    if X.std(axis=0).sum() == 0:
        return {"auc_missingness_only": 0.5, "note": "пропусков нет"}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    prob = cross_val_predict(LogisticRegression(max_iter=1000), X, df["label"], groups=df["patient_id"],
                             cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(df["label"], prob))
    return {"auc_missingness_only": auc,
            "warning": auc > 0.6,
            "columns": raw_cols}


def fig_samples(df, out, per_class: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(PAD_DIAGNOSES), per_class, figsize=(per_class * 1.7, len(PAD_DIAGNOSES) * 1.8))
    for i, d in enumerate(PAD_DIAGNOSES):
        part = df[df["diagnostic"] == d]
        pick = part.sample(min(per_class, len(part)), random_state=int(rng.integers(1e6))) if len(part) else part
        for j in range(per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(pick):
                img = Image.open(pick.iloc[j]["image_path"]).convert("RGB")
                img.thumbnail((256, 256))
                ax.imshow(img)
        axes[i, 0].set_title(f"{d}", loc="left", fontsize=10,
                             color=MALIGNANT if d in MALIGNANT_DEFAULT else BENIGN)
    fig.suptitle("Примеры снимков PAD-UFES-20 (смартфон)", x=0.02, ha="left", fontsize=12)
    return save(fig, out / "pad_samples.png")


def image_sizes(df, limit: int = 600) -> dict:
    sizes = []
    for p in df["image_path"].head(limit):
        with Image.open(p) as im:
            sizes.append(im.size)
    s = np.array(sizes)
    return {"checked": len(s), "min_side_median": int(np.median(s.min(1))),
            "min_side_min": int(s.min()), "max_side_max": int(s.max())}


def isic_eda(root: Path, out: Path) -> dict:
    from dermatriage.data.isic2024 import ISICHDF5Dataset, load_isic_metadata

    df = load_isic_metadata(root)
    pos_per_patient = df.groupby("patient_id")["label"].sum()
    res = {
        "images": len(df), "malignant": int(df["label"].sum()),
        "malignant_share": float(df["label"].mean()), "patients": int(df["patient_id"].nunique()),
        "patients_with_malignant": int((pos_per_patient > 0).sum()),
    }
    if "anatom_site_general" in df.columns:
        tab = pd.crosstab(df["anatom_site_general"].fillna("неизв."), df["label"])
        res["malignant_share_by_site"] = (tab[1] / tab.sum(axis=1)).round(5).to_dict() if 1 in tab else {}
    # примеры снимков
    ds = ISICHDF5Dataset(root / "train-image.hdf5", df["isic_id"], df["label"])
    rng = np.random.default_rng(0)
    pos = rng.choice(np.flatnonzero(df["label"] == 1), size=min(6, int(df["label"].sum())), replace=False)
    neg = rng.choice(np.flatnonzero(df["label"] == 0), size=6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
    for row, (idx, name, color) in enumerate(((neg, "доброкачественные", BENIGN), (pos, "злокачественные", MALIGNANT))):
        for j in range(6):
            ax = axes[row, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(ds.load_image(int(idx[j])))
        axes[row, 0].set_title(name, loc="left", fontsize=10, color=color)
    fig.suptitle("ISIC 2024 (SLICE-3D): фрагменты 3D-фото, качество близко к смартфону", x=0.02, ha="left")
    save(fig, out / "isic_samples.png")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default="data/pad_ufes_20")
    ap.add_argument("--isic", default="data/isic2024")
    ap.add_argument("--skip-isic", action="store_true")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()
    setup_style()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_pad_metadata(resolve(args.pad))
    service = {"image_path", "label"}
    raw_cols = [c for c in df.columns if c not in service and c not in
                ("patient_id", "lesion_id", "img_id", "diagnostic")]
    # для анализа пропусков берём исходные пропуски, а не нормализованные значения
    raw = pd.read_csv(next(resolve(args.pad).rglob("metadata.csv")))
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = raw.rename(columns={"fitspatrick": "fitzpatrick"})
    raw = raw[raw["img_id"].isin(df["img_id"])].set_index("img_id").loc[df["img_id"]].reset_index()
    for c in SYMPTOM_FIELDS:
        if c in raw:
            raw[c] = raw[c].where(~raw[c].astype(str).str.upper().isin(["UNK", "NAN"]))
    raw["diagnostic"] = df["diagnostic"].to_numpy()
    raw["label"] = df["label"].to_numpy()
    raw["patient_id"] = df["patient_id"].to_numpy()
    raw_cols = [c for c in raw_cols if c in raw.columns]

    fig_diagnoses(df, out)
    _, per = fig_images_per_patient(df, out)
    fig_age(df, out)
    _, sym = fig_symptoms(df, out)
    fig_regions(df, out)
    _, fitz = fig_fitzpatrick(df, out)
    _, miss = fig_missingness(raw, raw_cols, out)
    fig_samples(df, out)
    shortcut = shortcut_check(raw, raw_cols)
    sizes = image_sizes(df)

    summary = {
        "pad": {
            "images": len(df), "patients": int(df["patient_id"].nunique()),
            "lesions": int(df["lesion_id"].nunique()) if "lesion_id" in df else None,
            "diagnoses": df["diagnostic"].value_counts().to_dict(),
            "malignant_share": float(df["label"].mean()),
            "images_per_patient_max": int(per.max()),
            "patients_with_multiple_images": int((per > 1).sum()),
            "fitzpatrick_known": int(df["fitzpatrick"].notna().sum()),
            "image_sizes": sizes,
            "shortcut_check": shortcut,
        }
    }
    if not args.skip_isic:
        summary["isic"] = isic_eda(resolve(args.isic), out)
    save_json(summary, out / "eda_summary.json")

    p = summary["pad"]
    lines = [
        "# EDA: сводка", "",
        "Сгенерировано `scripts/eda.py`. Все числа посчитаны по данным, графики — в этой папке.", "",
        "## PAD-UFES-20", "",
        f"- Снимков: **{p['images']}**, пациентов: **{p['patients']}**, образований: {p['lesions']}.",
        f"- Доля злокачественных (BCC + MEL + SCC): **{p['malignant_share']:.1%}**.",
        f"- Пациентов с несколькими снимками: {p['patients_with_multiple_images']} "
        f"(максимум {p['images_per_patient_max']} на пациента) → разбиение по пациентам обязательно.",
        f"- Фототип известен для {p['fitzpatrick_known']} снимков.",
        f"- Размер снимков (проверено {sizes['checked']}): медиана короткой стороны "
        f"{sizes['min_side_median']} px, от {sizes['min_side_min']} до {sizes['max_side_max']} px.",
        f"- **Проверка «коротких путей»:** AUC модели только по факту пропусков в метаданных = "
        f"**{shortcut['auc_missingness_only']:.3f}**"
        + (" — ВНИМАНИЕ: способ заполнения анкеты связан с диагнозом; такие поля в модель не включаем."
           if shortcut.get("warning") else " — заметной связи нет."),
        "", "Диагнозы:", "",
        "| Диагноз | Снимков |", "| --- | --- |",
        *[f"| {d} — {PAD_DIAGNOSIS_NAMES_RU[d]} | {p['diagnoses'].get(d, 0)} |" for d in PAD_DIAGNOSES],
        "", "Фототип × метка:", "", fitz.rename(columns={0: "доброкач.", 1: "злокач."}).to_markdown(),
        "", "Анкета, доля «да» среди известных ответов и доля «неизвестно»:", "",
        sym.pivot(index="field", columns="label", values=["yes_share", "unk_share"]).round(3).to_markdown(),
        "", "![диагнозы](pad_diagnoses.png)", "![анкета](pad_symptoms.png)",
        "![пропуски](pad_missingness.png)", "![фототип](pad_fitzpatrick.png)",
        "![возраст](pad_age.png)", "![локализация](pad_regions.png)", "![примеры](pad_samples.png)",
    ]
    if "isic" in summary:
        s = summary["isic"]
        lines += ["", "## ISIC 2024 (SLICE-3D)", "",
                  f"- Снимков: **{s['images']}**, злокачественных: **{s['malignant']}** "
                  f"({s['malignant_share']:.3%}) — крайний дисбаланс.",
                  f"- Пациентов: {s['patients']}, из них со злокачественными: {s['patients_with_malignant']}.",
                  "", "![ISIC](isic_samples.png)"]
    (out / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA готов: {out / 'eda_summary.md'}")
    print(f"Проверка коротких путей: AUC по пропускам = {shortcut['auc_missingness_only']:.3f}")


if __name__ == "__main__":
    main()

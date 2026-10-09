"""Шаг 5. Глубокий анализ финальной модели: кривые обучения, ROC, матрица ошибок,
разбор по диагнозам и фототипам, эффект правила «красных флагов», галерея ошибок с Grad-CAM.

    python scripts/analyze.py --exp pad_mm_effb0_isic --compare pad_effb0 tab_logreg red_flags_rule

Результат: reports/analysis/<эксперимент>/analysis.md + графики.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from dermatriage.config import load_config
from dermatriage.data.pad_ufes import PAD_DIAGNOSES, PAD_DIAGNOSIS_NAMES_RU, red_flags
from dermatriage.experiments import label
from dermatriage.metrics import binary_report, safe_auc
from dermatriage.pipeline import prepare_pad, resolve
from dermatriage.plotting import BENIGN, MALIGNANT, SERIES, TEXT_2, save, setup_style
from dermatriage.utils import fmt_mean_std, load_json, save_json

import matplotlib.pyplot as plt  # noqa: E402

FITZ_GROUPS = {1: "I–II", 2: "I–II", 3: "III", 4: "IV–VI", 5: "IV–VI", 6: "IV–VI"}


def seeds_of(runs: Path, exp: str) -> list[int]:
    return sorted(int(p.name[4:]) for p in (runs / exp).glob("seed*") if (p / "preds_test.csv").exists())


def load_preds(runs: Path, exp: str, seed: int):
    d = runs / exp / f"seed{seed}"
    thr = load_json(d / "threshold.json")["threshold"] if (d / "threshold.json").exists() else \
        load_json(d / "metrics.json")["test"]["at_val_threshold"]["threshold"]
    return pd.read_csv(d / "preds_val.csv"), pd.read_csv(d / "preds_test.csv"), thr


# ---------------------------------------------------------------- графики
def plot_learning_curves(runs, exp, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, s in enumerate(seeds):
        h = runs / exp / f"seed{s}" / "history.csv"
        if not h.exists():
            continue
        h = pd.read_csv(h)
        c = SERIES[k % len(SERIES)]
        axes[0].plot(h["epoch"], h["train_loss"], color=c, label=f"train, seed {s}")
        axes[0].plot(h["epoch"], h["val_loss"], color=c, linestyle="--", label=f"val, seed {s}")
        axes[1].plot(h["epoch"], h["val_auc"], color=c, label=f"seed {s}")
        best = h.loc[h["val_auc"].idxmax()]
        axes[1].plot(best["epoch"], best["val_auc"], "o", color=c, markersize=8,
                     markeredgecolor="#fcfcfb", markeredgewidth=2)
    axes[0].set_title("Функция потерь")
    axes[0].set_xlabel("эпоха")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("AUC на валидации (точка — выбранная эпоха)")
    axes[1].set_xlabel("эпоха")
    axes[1].legend(fontsize=8)
    return save(fig, out / "learning_curves.png")


def plot_roc(runs, exps, seed, out):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle=":")
    for k, exp in enumerate(exps):
        try:
            _, te, thr = load_preds(runs, exp, seed)
        except FileNotFoundError:
            continue
        fpr, tpr, _ = roc_curve(te["label"], te["prob"])
        c = SERIES[k % len(SERIES)]
        ax.plot(fpr, tpr, color=c, label=f"{label(exp)} (AUC {safe_auc(te['label'], te['prob']):.3f})")
        r = binary_report(te["label"], te["prob"], thr)
        ax.plot(1 - r["specificity"], r["sensitivity"], "o", color=c, markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
    ax.axhline(0.9, color="#c3c2b7", linewidth=1, linestyle="--")
    ax.text(0.98, 0.885, "цель: чувствительность 90%", ha="right", va="top", fontsize=8, color=TEXT_2)
    ax.set_xlabel("1 − специфичность (доля лишних направлений среди здоровых)")
    ax.set_ylabel("чувствительность")
    ax.set_title(f"ROC на тесте (seed {seed}); точка — порог с валидации")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    return save(fig, out / "roc.png")


def plot_confusion(te, thr, out, title):
    r = binary_report(te["label"], te["prob"], thr)
    m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(m, cmap="Blues")
    ax.grid(False)
    names = [["верно: наблюдать", "лишнее направление"], ["ПРОПУСК", "верно: направить"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > m.max() * 0.6 else "#0b0b0b")
    ax.set_xticks([0, 1], ["наблюдать", "направить"])
    ax.set_yticks([0, 1], ["доброкач.", "злокач."])
    ax.set_xlabel("решение системы")
    ax.set_ylabel("истинный класс")
    ax.set_title(title)
    return save(fig, out / "confusion.png"), r


def plot_threshold_tradeoff(te, thr, out):
    ts = np.linspace(0, 1, 201)
    sens = [binary_report(te["label"], te["prob"], t)["sensitivity"] for t in ts]
    spec = [binary_report(te["label"], te["prob"], t)["specificity"] for t in ts]
    ref = [(te["prob"] >= t).mean() for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(ts, sens, color=MALIGNANT, label="чувствительность")
    ax.plot(ts, spec, color=BENIGN, label="специфичность")
    ax.plot(ts, ref, color=SERIES[2], label="доля направлений", linestyle="--")
    ax.axvline(thr, color="#52514e", linewidth=1)
    ax.text(thr + 0.01, 0.04, f"порог {thr:.2f}\n(с валидации)", fontsize=8, color=TEXT_2)
    ax.set_xlabel("порог вероятности")
    ax.set_ylim(0, 1.02)
    ax.set_title("Выбор порога: компромисс пропусков и лишних направлений (тест)")
    ax.legend(loc="center right")
    return save(fig, out / "threshold_tradeoff.png")


def per_group_table(runs, exp, seeds, group_fn, order=None) -> pd.DataFrame:
    """Метрики по подгруппам (диагноз, фототип…), среднее по seed + размер подгруппы."""
    rows = []
    for s in seeds:
        _, te, thr = load_preds(runs, exp, s)
        te = te.assign(group=group_fn(te), refer=(te["prob"] >= thr).astype(int))
        for g, part in te.groupby("group"):
            r = {"seed": s, "group": g, "n": len(part), "malignant": int(part["label"].sum()),
                 "referral_rate": part["refer"].mean(), "auc": safe_auc(part["label"], part["prob"])}
            pos, neg = part[part["label"] == 1], part[part["label"] == 0]
            r["sensitivity"] = pos["refer"].mean() if len(pos) else np.nan
            r["specificity"] = 1 - neg["refer"].mean() if len(neg) else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    agg = t.groupby("group").agg(n=("n", "sum"), malignant=("malignant", "sum"),
                                 referral_rate=("referral_rate", "mean"), sensitivity=("sensitivity", "mean"),
                                 specificity=("specificity", "mean"), auc=("auc", "mean"))
    if order:
        agg = agg.reindex([o for o in order if o in agg.index])
    return agg


def plot_per_diagnosis(tab, out):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    d = tab.reindex([x for x in PAD_DIAGNOSES if x in tab.index])
    colors = [MALIGNANT if x in ("BCC", "MEL", "SCC") else BENIGN for x in d.index]
    ax.barh([f"{x} — {PAD_DIAGNOSIS_NAMES_RU[x]} (n={int(d.loc[x, 'n'])})" for x in d.index],
            d["referral_rate"] * 100, color=colors, height=0.6)
    for i, v in enumerate(d["referral_rate"] * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=TEXT_2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% снимков, направленных к специалисту")
    ax.set_title("Кого система направляет: по диагнозам (тест, среднее по seed)")
    ax.grid(axis="y", visible=False)
    return save(fig, out / "per_diagnosis.png")


def plot_fairness(tab, out):
    groups = tab.index.tolist()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, tab["sensitivity"] * 100, width=w - 0.04, color=MALIGNANT, label="чувствительность")
    ax.bar(x + w / 2, tab["specificity"] * 100, width=w - 0.04, color=BENIGN, label="специфичность")
    ax.set_xticks(x, [f"{g}\n(n={int(tab.loc[g, 'n'])}, злок. {int(tab.loc[g, 'malignant'])})" for g in groups])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_xlabel("фототип по Фицпатрику")
    ax.set_title("Справедливость: качество по фототипу кожи (тест)")
    ax.legend(loc="lower right")
    return save(fig, out / "fairness_fitzpatrick.png")


def red_flag_effect(runs, exp, seeds) -> pd.DataFrame:
    rows = []
    for s in seeds:
        _, te, thr = load_preds(runs, exp, s)
        model = (te["prob"] >= thr).to_numpy()
        flags = red_flags(te) if all(c in te for c in ("bleed", "changed", "grew")) else np.zeros(len(te), bool)
        for name, dec in (("модель", model), ("модель ИЛИ красный флаг", model | flags)):
            r = binary_report(te["label"], dec.astype(float), 0.5)
            rows.append({"seed": s, "решение": name, **{k: r[k] for k in
                                                         ("sensitivity", "specificity", "referral_rate", "fn")}})
    t = pd.DataFrame(rows)
    return t.groupby("решение").agg(
        чувствительность=("sensitivity", lambda v: fmt_mean_std(v.tolist())),
        специфичность=("specificity", lambda v: fmt_mean_std(v.tolist())),
        доля_направлений=("referral_rate", lambda v: fmt_mean_std(v.tolist())),
        пропусков_всего=("fn", "sum"))


def error_gallery(runs, exp, seed, out, k: int = 6):
    """Самые уверенные ошибки модели с Grad-CAM — основа разбора ошибок на защите."""
    from dermatriage.inference import TriageModel

    run_dir = runs / exp / f"seed{seed}"
    cfg = load_config(run_dir / "config.yaml")
    df = prepare_pad(cfg, seed)
    _, te, thr = load_preds(runs, exp, seed)
    extra = [c for c in ("image_path", "age", "gender", "region", "itch", "grew", "hurt", "changed",
                         "bleed", "elevation") if c in df.columns and c not in te.columns]
    te = te.merge(df[["img_id"] + extra], on="img_id", how="left")
    tm = TriageModel(run_dir, use_red_flags=False)
    sets = {
        "fn": ("Пропуски: злокачественные с самой низкой оценкой", te[te["label"] == 1].nsmallest(k, "prob")),
        "fp": ("Лишние направления: доброкачественные с самой высокой оценкой", te[te["label"] == 0].nlargest(k, "prob")),
        "tp": ("Уверенно верные: злокачественные с самой высокой оценкой", te[te["label"] == 1].nlargest(k, "prob")),
    }
    paths = {}
    for key, (title, part) in sets.items():
        if part.empty:
            continue
        fig, axes = plt.subplots(2, len(part), figsize=(2.1 * len(part), 4.6), squeeze=False)
        for j, (_, row) in enumerate(part.iterrows()):
            answers = {c: row.get(c) for c in ("age", "gender", "region", "itch", "grew", "hurt",
                                                "changed", "bleed", "elevation")}
            res = tm.predict(row["image_path"], answers, explain=True)
            from dermatriage.data.datasets import load_rgb
            img = load_rgb(row["image_path"], 320)
            axes[0, j].imshow(img)
            axes[1, j].imshow(res.cam_overlay.resize(img.size))
            for i in (0, 1):
                axes[i, j].axis("off")
            axes[0, j].set_title(f"{row['diagnostic']}  p={row['prob']:.2f}", fontsize=9,
                                 color=MALIGNANT if row["label"] == 1 else BENIGN)
        fig.suptitle(f"{title} (порог {thr:.2f})", x=0.02, ha="left", fontsize=11)
        paths[key] = save(fig, out / f"errors_{key}.png")
        part[["img_id", "patient_id", "diagnostic", "prob", "fitzpatrick", "region"]].to_csv(
            out / f"errors_{key}.csv", index=False)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="pad_mm_effb0_isic")
    ap.add_argument("--compare", nargs="*", default=["pad_effb0", "tab_logreg", "red_flags_rule"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0, help="seed для ROC и галереи ошибок")
    ap.add_argument("--out", default="reports/analysis")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    setup_style()
    runs = resolve(args.runs)
    out = resolve(args.out) / args.exp
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds_of(runs, args.exp)
    if not seeds:
        raise SystemExit(f"Нет запусков {runs / args.exp}")

    plot_learning_curves(runs, args.exp, seeds, out)
    plot_roc(runs, [args.exp] + args.compare, args.seed, out)
    _, te, thr = load_preds(runs, args.exp, args.seed)
    _, conf = plot_confusion(te, thr, out, f"Матрица ошибок (тест, seed {args.seed})")
    plot_threshold_tradeoff(te, thr, out)

    diag = per_group_table(runs, args.exp, seeds, lambda t: t["diagnostic"], PAD_DIAGNOSES)
    plot_per_diagnosis(diag, out)
    fitz = per_group_table(runs, args.exp, seeds,
                           lambda t: t["fitzpatrick"].map(lambda v: FITZ_GROUPS.get(int(v), "неизв.")
                                                          if pd.notna(v) else "неизв."),
                           ["I–II", "III", "IV–VI", "неизв."])
    plot_fairness(fitz, out)
    flags = red_flag_effect(runs, args.exp, seeds)
    gallery = {} if args.no_gallery else error_gallery(runs, args.exp, args.seed, out)

    fmt = lambda t: t.round(3).to_markdown()  # noqa: E731
    lines = [
        f"# Анализ: {label(args.exp)}", "",
        f"Запусков (seed): {seeds}. Сгенерировано `scripts/analyze.py`.", "",
        "## Кривые обучения", "", "![](learning_curves.png)", "",
        "## ROC и выбранный порог", "", "![](roc.png)", "", "![](threshold_tradeoff.png)", "",
        f"## Матрица ошибок (seed {args.seed}, порог {thr:.3f})", "",
        f"Чувствительность {conf['sensitivity']:.3f}, специфичность {conf['specificity']:.3f}, "
        f"пропущено злокачественных: {conf['fn']}, лишних направлений: {conf['fp']}.", "",
        "![](confusion.png)", "",
        "## По диагнозам: какая доля направлена к специалисту", "",
        "Для злокачественных (BCC, MEL, SCC) это чувствительность по подтипу; для ACK (предрак) "
        "направление клинически оправдано; для NEV и SEK — это лишние направления.", "",
        fmt(diag), "", "![](per_diagnosis.png)", "",
        "## Справедливость: по фототипу кожи", "",
        "Маленькие подгруппы дают неустойчивые оценки — смотрите на n.", "",
        fmt(fitz), "", "![](fairness_fitzpatrick.png)", "",
        "## Правило безопасности «красных флагов»", "",
        "Направить, если модель считает риск высоким ИЛИ в анкете: кровоточит / изменялось / растёт.", "",
        flags.to_markdown(), "",
    ]
    if gallery:
        lines += ["## Разбор ошибок (Grad-CAM)", ""]
        for key in ("fn", "fp", "tp"):
            if key in gallery:
                lines += [f"![](errors_{key}.png)", ""]
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    save_json({"per_diagnosis": diag.reset_index().to_dict("records"),
               "fairness": fitz.reset_index().to_dict("records"),
               "confusion_seed": conf}, out / "analysis.json")
    print(f"Анализ готов: {out / 'analysis.md'}")


if __name__ == "__main__":
    main()

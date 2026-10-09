"""Шаг 6. Сводная таблица всех экспериментов (среднее ± ст. отклонение по seed).

    python scripts/collect_results.py --runs runs --out reports/results

Результат: results.md (таблица для отчёта и слайдов), results.csv, results_auc.png.
Все числа читаются из runs/*/seed*/metrics.json — вручную ничего не вписывается.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from dermatriage.experiments import EXPERIMENT_GROUP, label
from dermatriage.pipeline import resolve
from dermatriage.plotting import SERIES, TEXT_2, save, setup_style
from dermatriage.utils import fmt_mean_std, load_json

import matplotlib.pyplot as plt  # noqa: E402

METRICS = [
    ("Val AUC", lambda m: m["val"]["auc"]),
    ("AUC", lambda m: m["test"]["auc"]),
    ("pAUC80", lambda m: m["test"]["pauc80"]),
    ("Чувствительность", lambda m: m["test"]["at_val_threshold"]["sensitivity"]),
    ("Специфичность", lambda m: m["test"]["at_val_threshold"]["specificity"]),
    ("Balanced acc.", lambda m: m["test"]["at_val_threshold"]["balanced_accuracy"]),
    ("Доля направлений", lambda m: m["test"]["at_val_threshold"]["referral_rate"]),
]


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(runs.glob("*/seed*/metrics.json")):
        m = load_json(mpath)
        if "test" not in m or "at_val_threshold" not in m.get("test", {}):
            continue  # предобучение ISIC — отдельная таблица
        row = {"experiment": m.get("experiment", mpath.parent.parent.name), "seed": m.get("seed")}
        for name, fn in METRICS:
            try:
                row[name] = fn(m)
            except (KeyError, TypeError):
                row[name] = np.nan
        row["params_m"] = m.get("model", {}).get("params_millions")
        lat = m.get("latency_ms_batch1", {})
        row["latency_ms"] = next(iter(lat.values())) if lat else None
        row["train_seconds"] = m.get("training", {}).get("train_seconds")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for exp, g in df.groupby("experiment"):
        row = {"Модель": label(exp), "Группа": EXPERIMENT_GROUP.get(exp, ""), "seeds": len(g),
               "_val_auc": g["Val AUC"].mean(), "_exp": exp}
        for name, _ in METRICS:
            row[name] = fmt_mean_std(g[name].tolist())
        p = g["params_m"].dropna()
        row["Параметры, млн"] = f"{p.iloc[0]:.1f}" if len(p) else "—"
        lat = g["latency_ms"].dropna()
        row["Задержка, мс"] = f"{lat.mean():.1f}" if len(lat) else "—"
        out.append(row)
    # Модели ранжируются по ВАЛИДАЦИИ: выбирать финальную модель по тесту — подглядывание.
    return pd.DataFrame(out).sort_values("_val_auc", ascending=False)


def paired_differences(df: pd.DataFrame, best: str) -> pd.DataFrame:
    """Разница AUC «лучшая модель − другая» на одних и тех же разбиениях (seed)."""
    piv = df.pivot_table(index="seed", columns="experiment", values="AUC")
    rows = []
    for exp in piv.columns:
        if exp == best:
            continue
        d = (piv[best] - piv[exp]).dropna()
        if len(d):
            rows.append({"Сравнение": f"{label(best)} − {label(exp)}", "ΔAUC (по seed)": fmt_mean_std(d.tolist()),
                         "лучше в seed": f"{int((d > 0).sum())} из {len(d)}"})
    return pd.DataFrame(rows)


def plot_auc(df: pd.DataFrame, order: list[str], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(order) + 1.2))
    groups = ["baseline", "анкета", "фото", "фото + анкета"]
    colors = {g: SERIES[i] for i, g in enumerate(groups)}
    for y, exp in enumerate(order[::-1]):
        vals = df.loc[df["experiment"] == exp, "AUC"].dropna()
        if vals.empty:
            continue
        c = colors.get(EXPERIMENT_GROUP.get(exp, ""), SERIES[4])
        ax.plot([vals.min(), vals.max()], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(vals.mean(), y, "o", color=c, markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.text(vals.max() + 0.006, y, f"{vals.mean():.3f}", va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(range(len(order)), [label(e) for e in order[::-1]])
    ax.set_xlabel("AUC на тесте (точка — среднее, линия — разброс по seed); порядок — по валидации")
    ax.set_title("Сравнение моделей уровня 1")
    ax.grid(axis="y", visible=False)
    lo = df["AUC"].min()
    ax.set_xlim(max(0.4, lo - 0.05), 1.04)
    from matplotlib.lines import Line2D
    present = [g for g in groups if any(EXPERIMENT_GROUP.get(e) == g for e in order)]
    ax.legend(handles=[Line2D([], [], marker="o", color=colors[g], linestyle="", label=g) for g in present],
              loc="lower right")
    return save(fig, out / "results_auc.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/results")
    ap.add_argument("--best", default=None,
                    help="эксперимент, с которым сравнивать остальные (по умолчанию — лучший по валидации)")
    args = ap.parse_args()
    setup_style()
    runs, out = resolve(args.runs), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = collect(runs)
    if df.empty:
        raise SystemExit(f"В {runs} нет завершённых запусков с metrics.json")
    df.to_csv(out / "results_per_seed.csv", index=False)
    table = summary_table(df)
    best = args.best or table.iloc[0]["_exp"]
    order = table["_exp"].tolist()
    plot_auc(df, order, out)
    table.drop(columns=["_val_auc", "_exp"]).to_csv(out / "results.csv", index=False)
    diffs = paired_differences(df, best)

    lines = ["# Результаты экспериментов (тестовая выборка)", "",
             "Сгенерировано `scripts/collect_results.py` из `runs/*/seed*/metrics.json`. "
             "Среднее ± стандартное отклонение по разным разбиениям по пациентам (seed). "
             "Чувствительность, специфичность и доля направлений — при пороге, подобранном на валидации "
             "под чувствительность ≥ 90%. Строки упорядочены по AUC на ВАЛИДАЦИИ (по ней выбирается "
             "финальная модель); тестовые столбцы — независимая оценка.", "",
             table.drop(columns=["_val_auc", "_exp"]).to_markdown(index=False), "",
             f"## Парные сравнения с «{label(best)}»", "",
             diffs.to_markdown(index=False) if len(diffs) else "Нет других экспериментов с теми же seed.", "",
             "![AUC](results_auc.png)"]

    isic = sorted(runs.glob("isic_pretrain*/seed*/metrics.json"))
    if isic:
        lines += ["", "## Предобучение на ISIC 2024 (SLICE-3D)", "",
                  "| Эксперимент | Выборка | Снимков | Злокач. | AUC | pAUC80 (макс. 0,2) | 95% ДИ pAUC80 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for p in isic:
            m = load_json(p)
            for s in ("val", "test"):
                r = m[s]
                ci = r.get("pauc80_ci95", [None, None])
                ci_s = f"[{ci[0]:.3f}; {ci[1]:.3f}]" if ci[0] is not None else "—"
                lines.append(f"| {m['experiment']} | {s} | {r['images']} | {r['malignant']} | "
                             f"{r['auc']:.3f} | {r['pauc80']:.3f} | {ci_s} |")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nГотово: {out / 'results.md'}")


if __name__ == "__main__":
    main()

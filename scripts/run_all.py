"""Запускает весь план экспериментов по порядку. Уже готовые шаги пропускаются,
поэтому после сбоя или перезагрузки достаточно запустить команду ещё раз.

    python scripts/run_all.py                      # полный план (несколько часов на GPU)
    python scripts/run_all.py --quick              # пробный прогон: 2 эпохи, 1 seed
    python scripts/run_all.py --skip-isic          # без предобучения на ISIC 2024
    python scripts/run_all.py --only pad_mm_effb0  # только выбранные эксперименты

План:
  0. проверка данных → 1. EDA → 2. калибровка качества снимка → 3. табличные baseline
  → 4. предобучение на ISIC 2024 → 5. модели на PAD-UFES-20 → 6. анализ → 7. сводная таблица
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAD_EXPERIMENTS = [
    "pad_effb0",          # фото, ImageNet — основной image-baseline
    "pad_resnet50",       # фото, другая CNN
    "pad_vit_b16",        # фото, трансформер
    "pad_effb0_isic",     # фото, предобучение ISIC 2024
    "pad_mm_effb0",       # фото + анкета, ImageNet
    "pad_mm_effb0_isic",  # фото + анкета, ISIC 2024 (кандидат в финальную модель)
]
NEEDS_ISIC = {"pad_effb0_isic", "pad_mm_effb0_isic"}


def run(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 80}\n▶ {title}\n  {' '.join(cmd)}\n{'=' * 80}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"Шаг «{title}» завершился с ошибкой (код {r.returncode}). "
                         "Исправьте и запустите run_all.py снова — готовые шаги будут пропущены.")
    print(f"✓ {title}: {(time.time() - t0) / 60:.1f} мин", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="2 эпохи и 1 seed — проверить, что всё запускается")
    ap.add_argument("--skip-isic", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--final", default="pad_mm_effb0_isic", help="эксперимент для подробного анализа")
    ap.add_argument("--pad", default="data/pad_ufes_20")
    ap.add_argument("--isic", default="data/isic2024")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--quality-out", default="configs/quality_limits.json")
    ap.add_argument("--set", nargs="*", default=[], help="переопределения обучения для всех шагов, key=value")
    args = ap.parse_args()

    seeds = [args.seeds[0]] if args.quick else args.seeds
    reports = Path(args.reports) if Path(args.reports).is_absolute() else ROOT / args.reports
    runs = Path(args.runs) if Path(args.runs).is_absolute() else ROOT / args.runs
    quality = Path(args.quality_out) if Path(args.quality_out).is_absolute() else ROOT / args.quality_out
    data_sets = [f"data.pad_root={args.pad}", f"output.runs_dir={args.runs}"]
    train_sets = data_sets + args.set + (["train.epochs=2"] if args.quick else [])
    seed_args = ["--seeds", *map(str, seeds)]
    skip = ["--skip-isic"] if args.skip_isic else []

    run(["scripts/check_data.py", "--pad", args.pad, "--isic", args.isic] + skip, "0. Проверка данных")
    if not (reports / "eda/eda_summary.md").exists():
        run(["scripts/eda.py", "--pad", args.pad, "--isic", args.isic, "--out", str(reports / "eda")] + skip,
            "1. EDA")
    if not quality.exists():
        run(["scripts/calibrate_quality.py", "--out", str(quality), "--fig-dir", str(reports / "quality"),
             "--set", f"data.pad_root={args.pad}"], "2. Калибровка проверки качества снимка")
    run(["scripts/baselines_tabular.py", *seed_args, "--set", *data_sets], "3. Базовые линии по анкете")

    exps = args.only or PAD_EXPERIMENTS
    final = args.final
    isic_enc = runs / "isic_pretrain_efficientnet_b0/seed0/encoder.pt"
    if args.skip_isic:
        exps = [e for e in exps if e not in NEEDS_ISIC]
        if final in NEEDS_ISIC:
            final = "pad_mm_effb0"
    elif any(e in NEEDS_ISIC for e in exps):
        if isic_enc.exists():
            print(f"\nПредобучение ISIC уже выполнено: {isic_enc}")
        else:
            isic_sets = [f"data.isic_root={args.isic}", f"output.runs_dir={args.runs}"] + args.set
            if args.quick:
                isic_sets += ["train.epochs=2", "data.val_neg_max=2000"]
            run(["scripts/pretrain_isic.py", "--set", *isic_sets], "4. Предобучение на ISIC 2024")

    for e in exps:
        extra = [f"model.isic_encoder={isic_enc}"] if e in NEEDS_ISIC else []
        run(["scripts/train_pad.py", "--config", f"configs/{e}.yaml", "--skip-existing", *seed_args,
             "--set", *train_sets, *extra], f"5. Обучение {e}")

    if (runs / final).exists():
        compare = [e for e in ("pad_effb0", "pad_mm_effb0", "tab_logreg", "red_flags_rule")
                   if e != final and (runs / e).exists()]
        run(["scripts/analyze.py", "--exp", final, "--compare", *compare, "--seed", str(seeds[0]),
             "--runs", args.runs, "--out", str(reports / "analysis")], f"6. Анализ {final}")
    run(["scripts/collect_results.py", "--best", final, "--runs", args.runs, "--out", str(reports / "results")],
        "7. Сводная таблица")
    print(f"\nГотово. Смотрите {reports / 'results/results.md'} и {reports / 'analysis'}.")


if __name__ == "__main__":
    main()

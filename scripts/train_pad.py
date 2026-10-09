"""Шаг 4. Обучение модели уровня 1 на PAD-UFES-20 (только фото или фото + анкета).

    python scripts/train_pad.py --config configs/pad_mm_effb0_isic.yaml --seeds 0 1 2

Для каждого seed своё разбиение по пациентам. В runs/<эксперимент>/seed<k>/ сохраняются:
  best.pt          веса лучшей эпохи (по AUC на валидации)
  history.csv      кривые обучения
  preds_val.csv, preds_test.csv  вероятности для каждого снимка
  threshold.json   порог под чувствительность ≥ 90%, подобранный на валидации
  tab_encoder.json кодировщик анкеты (для мультимодальной модели)
  metrics.json     итоговые метрики + сведения об окружении
  config.yaml      полный конфиг запуска
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from dermatriage.config import load_config, save_config
from dermatriage.engine import predict, train_model
from dermatriage.metrics import evaluate_split_predictions, threshold_for_sensitivity
from dermatriage.models.networks import build_model, count_parameters, load_encoder_weights
from dermatriage.pipeline import (experiment_name, fit_tab_encoder, make_pad_loaders, predictions_frame,
                                  prepare_pad, resolve, run_dir_for)
from dermatriage.utils import env_info, get_device, save_json, set_seed


@torch.no_grad()
def measure_latency(model, device, size: int, tab_dim: int, n: int = 30) -> float:
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    tab = torch.zeros(1, tab_dim, device=device) if tab_dim else None
    for _ in range(5):
        model(x, tab)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x, tab)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def run_seed(cfg: dict, name: str, seed: int) -> dict:
    set_seed(seed, cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    run_dir = run_dir_for(cfg, name, seed)
    print(f"\n=== {name} | seed {seed} | {device} → {run_dir}")

    df = prepare_pad(cfg, seed)
    mcfg = cfg["model"]
    multimodal = mcfg.get("kind") == "multimodal"
    enc = fit_tab_encoder(cfg, df[df["split"] == "train"]) if multimodal else None
    tab_dim = enc.dim if enc else 0

    model = build_model(mcfg, tab_dim=tab_dim, img_size=cfg["data"]["img_size"])
    if mcfg.get("init") == "isic":
        path = resolve(mcfg.get("isic_encoder"))
        if path is None or not path.exists():
            raise FileNotFoundError(f"Нет весов предобучения ISIC: {path}. Сначала запустите scripts/pretrain_isic.py")
        load_encoder_weights(model, path)
        print(f"Энкодер инициализирован весами ISIC 2024: {path}")
    print(f"Параметров: {count_parameters(model) / 1e6:.2f} млн, анкета: {tab_dim} признаков")

    loaders = make_pad_loaders(cfg, df, enc)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    t0 = time.time()
    history = train_model(model, loaders["train"], loaders["val"], va["label"].to_numpy(), cfg["train"],
                          device, run_dir, train_labels=tr["label"].to_numpy())
    train_sec = time.time() - t0

    tta = cfg["eval"].get("tta", True)
    pv = predictions_frame(va, predict(model, loaders["val"], device, tta=tta))
    pt = predictions_frame(te, predict(model, loaders["test"], device, tta=tta))
    pv.to_csv(run_dir / "preds_val.csv", index=False)
    pt.to_csv(run_dir / "preds_test.csv", index=False)

    target = cfg["eval"]["target_sensitivity"]
    thr = threshold_for_sensitivity(pv["label"], pv["prob"], target)
    save_json({"threshold": thr, "target_sensitivity": target, "chosen_on": "val", "tta": tta},
              run_dir / "threshold.json")
    if enc is not None:
        save_json(enc.to_dict(), run_dir / "tab_encoder.json")

    metrics = evaluate_split_predictions(pv, pt, target, cfg["eval"]["n_boot"])
    best_row = history.loc[history["val_auc"].idxmax()] if history["val_auc"].notna().any() else history.iloc[-1]
    metrics.update({
        "experiment": name, "seed": seed,
        "model": {"kind": mcfg.get("kind"), "backbone": mcfg["backbone"], "init": mcfg.get("init"),
                  "params_millions": count_parameters(model) / 1e6,
                  "size_mb": (run_dir / "best.pt").stat().st_size / 2 ** 20,
                  "tab_dim": tab_dim},
        "training": {"epochs_run": int(len(history)), "best_epoch": int(best_row["epoch"]),
                     "train_seconds": round(train_sec, 1)},
        "latency_ms_batch1": {device.type: round(measure_latency(model, device, cfg["data"]["img_size"], tab_dim), 2)},
        "split_sizes": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "env": env_info(),
    })
    save_json(metrics, run_dir / "metrics.json")
    save_config({**cfg, "experiment": name}, run_dir / "config.yaml")
    t = metrics["test"]
    print(f"ТЕСТ: AUC {t['auc']:.3f} [{t['auc_ci95'][0]:.3f}; {t['auc_ci95'][1]:.3f}]  pAUC80 {t['pauc80']:.3f}  "
          f"при пороге {thr:.3f}: чувствительность {t['at_val_threshold']['sensitivity']:.3f}, "
          f"специфичность {t['at_val_threshold']['specificity']:.3f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=[], help="переопределения key=value, напр. train.epochs=5")
    ap.add_argument("--skip-existing", action="store_true", help="пропускать seed, для которых уже есть metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    name = experiment_name(cfg, args.config)
    aucs = []
    for s in args.seeds:
        done = resolve(cfg["output"]["runs_dir"]) / name / f"seed{s}" / "metrics.json"
        if args.skip_existing and done.exists():
            print(f"{name} seed {s}: уже обучено, пропускаю ({done})")
            from dermatriage.utils import load_json
            aucs.append(load_json(done)["test"]["auc"])
            continue
        aucs.append(run_seed(cfg, name, s)["test"]["auc"])
    print(f"\n{name}: test AUC по seed = {np.round(aucs, 3).tolist()}, среднее {np.nanmean(aucs):.3f}")


if __name__ == "__main__":
    main()

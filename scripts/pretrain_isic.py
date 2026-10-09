"""Шаг 3. Предобучение энкодера на ISIC 2024 (SLICE-3D) — 400 000+ снимков смартфонного качества.

    python scripts/pretrain_isic.py --config configs/isic_pretrain.yaml

Идея: PAD-UFES-20 маленький (≈2 300 снимков). Сначала учим энкодер отличать
злокачественные образования на большом наборе того же «смартфонного» типа,
затем дообучаем на PAD-UFES-20. Эффект проверяется экспериментом
pad_effb0 (ImageNet) против pad_effb0_isic (ImageNet → ISIC 2024).

Результат: runs/isic_pretrain_efficientnet_b0/seed0/encoder.pt (+ метрики на val/test ISIC).
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from dermatriage.config import load_config, save_config
from dermatriage.data.isic2024 import BalancedNegativeSampler, ISICHDF5Dataset, load_isic_metadata
from dermatriage.data.splits import patient_split, split_summary
from dermatriage.data.transforms import build_eval_transform, build_train_transform
from dermatriage.engine import predict, train_model
from dermatriage.metrics import bootstrap_ci, pauc_above_tpr, safe_auc
from dermatriage.models.networks import build_model, count_parameters
from dermatriage.pipeline import experiment_name, resolve, run_dir_for
from dermatriage.utils import env_info, get_device, save_json, set_seed


def subsample_negatives(part, max_neg: int, seed: int):
    pos = part[part["label"] == 1]
    neg = part[part["label"] == 0]
    if len(neg) > max_neg:
        neg = neg.sample(max_neg, random_state=seed)
    return np.concatenate([pos.index.to_numpy(), neg.index.to_numpy()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isic_pretrain.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed, cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    name = experiment_name(cfg, args.config)
    run_dir = run_dir_for(cfg, name, seed)
    dcfg, tcfg = cfg["data"], cfg["train"]

    root = resolve(dcfg["isic_root"])
    df = load_isic_metadata(root)
    df["split"] = patient_split(df, seed=seed)
    print(split_summary(df, df["split"]).to_string(index=False))
    h5 = root / "train-image.hdf5"

    tr = df[df["split"] == "train"].reset_index(drop=True)
    va_idx = subsample_negatives(df[df["split"] == "val"], dcfg["val_neg_max"], seed)
    te_idx = subsample_negatives(df[df["split"] == "test"], dcfg["val_neg_max"], seed)
    va, te = df.loc[va_idx].reset_index(drop=True), df.loc[te_idx].reset_index(drop=True)

    size = dcfg["img_size"]
    nw = dcfg.get("num_workers", 4)
    train_ds = ISICHDF5Dataset(h5, tr["isic_id"], tr["label"], build_train_transform(size, dcfg.get("augment", "phone")))
    sampler = BalancedNegativeSampler(tr["label"], neg_per_pos=dcfg["neg_per_pos"], seed=seed)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, num_workers=nw,
                              pin_memory=torch.cuda.is_available(), drop_last=len(sampler) > tcfg["batch_size"],
                              persistent_workers=nw > 0)

    def eval_loader(part):
        ds = ISICHDF5Dataset(h5, part["isic_id"], part["label"], build_eval_transform(size))
        return DataLoader(ds, batch_size=tcfg["batch_size"] * 2, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    val_loader, test_loader = eval_loader(va), eval_loader(te)
    print(f"Эпоха: {len(sampler)} снимков ({len(sampler.pos)} злокачественных). "
          f"Валидация: {len(va)} ({int(va['label'].sum())} злокач.), тест: {len(te)} ({int(te['label'].sum())} злокач.)")

    model = build_model(cfg["model"], img_size=dcfg["img_size"])
    print(f"Модель {cfg['model']['backbone']}: {count_parameters(model) / 1e6:.2f} млн параметров, устройство {device}")
    hist = train_model(model, train_loader, val_loader, va["label"].to_numpy(), tcfg, device, run_dir,
                       train_labels=tr["label"].to_numpy())

    torch.save(model.encoder.state_dict(), run_dir / "encoder.pt")
    res = {"experiment": name, "seed": seed, "backbone": cfg["model"]["backbone"]}
    for split, part, loader in (("val", va, val_loader), ("test", te, test_loader)):
        prob = predict(model, loader, device, tta=False)
        part[["isic_id", "patient_id", "label"]].assign(prob=prob).to_csv(run_dir / f"preds_{split}.csv", index=False)
        lo, hi = bootstrap_ci(part["label"].to_numpy(), prob, groups=part["patient_id"].to_numpy(),
                              metric=pauc_above_tpr, n_boot=cfg["eval"].get("n_boot", 500), seed=seed)
        res[split] = {"auc": safe_auc(part["label"], prob), "pauc80": pauc_above_tpr(part["label"], prob),
                      "pauc80_ci95": [lo, hi], "images": len(part), "malignant": int(part["label"].sum())}
    res.update({"epochs_run": len(hist), "best_epoch": int(hist.loc[hist["val_pauc80"].idxmax(), "epoch"])
                if hist["val_pauc80"].notna().any() else None, "env": env_info()})
    save_json(res, run_dir / "metrics.json")
    save_config(cfg, run_dir / "config.yaml")
    print(f"\nISIC 2024 тест: AUC {res['test']['auc']:.3f}, pAUC80 {res['test']['pauc80']:.3f} (максимум 0,2)")
    print(f"Веса энкодера: {run_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()

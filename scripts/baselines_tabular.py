"""Шаг 2. Базовые линии без нейросетей — с ними сравниваются все остальные модели.

    python scripts/baselines_tabular.py --config configs/pad_base.yaml --seeds 0 1 2

Модели:
  prior          — константа (доля злокачественных на train). AUC = 0,5 по определению.
  red_flags_rule — клиническое правило: число «красных флагов» (кровоточит, менялось, растёт).
  tab_logreg     — логистическая регрессия по анкете.
  tab_lgbm       — градиентный бустинг (LightGBM) по анкете.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dermatriage.config import load_config, save_config
from dermatriage.data.pad_ufes import red_flags
from dermatriage.metrics import binary_report, evaluate_split_predictions, safe_auc
from dermatriage.pipeline import fit_tab_encoder, predictions_frame, prepare_pad, run_dir_for
from dermatriage.utils import env_info, save_json, set_seed


def save_run(cfg, name, seed, val, test, extra=None):
    d = run_dir_for(cfg, name, seed)
    val.to_csv(d / "preds_val.csv", index=False)
    test.to_csv(d / "preds_test.csv", index=False)
    metrics = evaluate_split_predictions(val, test, cfg["eval"]["target_sensitivity"], cfg["eval"]["n_boot"])
    metrics.update({"experiment": name, "seed": seed, "env": env_info(), **(extra or {})})
    save_json(metrics, d / "metrics.json")
    save_config({**cfg, "experiment": name}, d / "config.yaml")
    t = metrics["test"]
    print(f"  {name:<16} test AUC {t['auc']:.3f}  pAUC80 {t['pauc80']:.3f}  "
          f"sens {t['at_val_threshold']['sensitivity']:.3f}  spec {t['at_val_threshold']['specificity']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pad_base.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=[], help="переопределения key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n=== seed {seed}")
        df = prepare_pad(cfg, seed)
        tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
        enc = fit_tab_encoder(cfg, tr)
        Xtr, Xva, Xte = enc.transform(tr), enc.transform(va), enc.transform(te)

        # 1. константа
        prior = float(tr["label"].mean())
        save_run(cfg, "prior", seed, predictions_frame(va, np.full(len(va), prior)),
                 predictions_frame(te, np.full(len(te), prior)))

        # 2. правило красных флагов: 0..3 флага → «вероятность» 0..1
        def flags_score(part):
            return sum((part[f] == "yes").astype(float) for f in ("bleed", "changed", "grew")).to_numpy() / 3

        save_run(cfg, "red_flags_rule", seed, predictions_frame(va, flags_score(va)),
                 predictions_frame(te, flags_score(te)),
                 {"note": "оценка = доля красных флагов (0, 1/3, 2/3, 1)",
                  "any_flag_test": binary_report(te["label"], red_flags(te).astype(float), 0.5)})

        # 3. логистическая регрессия: C подбирается по AUC на валидации
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, tr["label"])
            auc = safe_auc(va["label"], m.predict_proba(Xva)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, C, m)
        _, C, lr = best
        save_run(cfg, "tab_logreg", seed, predictions_frame(va, lr.predict_proba(Xva)[:, 1]),
                 predictions_frame(te, lr.predict_proba(Xte)[:, 1]), {"best_C": C})
        coef = pd.DataFrame({"feature": enc.feature_names, "coef": lr.coef_[0]})
        coef.sort_values("coef", key=np.abs, ascending=False).to_csv(
            run_dir_for(cfg, "tab_logreg", seed) / "coefficients.csv", index=False)

        # 4. LightGBM с ранней остановкой по валидации
        gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 reg_lambda=1.0, random_state=seed, verbose=-1)
        gbm.fit(Xtr, tr["label"], eval_X=(Xva,), eval_y=(va["label"],), eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        save_run(cfg, "tab_lgbm", seed, predictions_frame(va, gbm.predict_proba(Xva)[:, 1]),
                 predictions_frame(te, gbm.predict_proba(Xte)[:, 1]),
                 {"best_iteration": int(gbm.best_iteration_ or 0)})


if __name__ == "__main__":
    main()

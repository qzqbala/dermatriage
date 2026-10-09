"""Автотесты: проверяют корректность ключевых частей и что весь пайплайн запускается.

    pytest -q                    # быстрые тесты (~1 мин на CPU)
    pytest -q -m slow            # + сквозной прогон run_all.py на синтетике (~3–5 мин на CPU)

Используются СИНТЕТИЧЕСКИЕ данные той же структуры, что PAD-UFES-20 и ISIC 2024.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dermatriage.data.pad_ufes import load_pad_metadata, normalize_ternary  # noqa: E402
from dermatriage.data.splits import patient_split  # noqa: E402
from dermatriage.data.tabular import TabularEncoder  # noqa: E402
from dermatriage.explain import grad_cam  # noqa: E402
from dermatriage.metrics import (binary_report, pauc_above_tpr, safe_auc,  # noqa: E402
                                 threshold_for_sensitivity)
from dermatriage.models.networks import ImageClassifier, MultimodalClassifier  # noqa: E402

TINY = ["data.img_size=64", "data.num_workers=0", "train.batch_size=16", "model.pretrained=false",
        "eval.n_boot=20"]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    subprocess.run([sys.executable, "scripts/make_synthetic_data.py", "--out", str(out), "--n-patients", "90",
                    "--n-isic", "300"], cwd=ROOT, check=True)
    return out


# ---------------------------------------------------------------- данные
def test_ternary_normalization():
    assert normalize_ternary("True") == "yes"
    assert normalize_ternary(False) == "no"
    assert normalize_ternary("UNK") == "unk"
    assert normalize_ternary(float("nan")) == "unk"


def test_patient_split_has_no_leakage(synthetic):
    df = load_pad_metadata(synthetic / "pad_ufes_20", verbose=False)
    for seed in (0, 1, 2):
        split = patient_split(df, seed=seed)
        groups = {s: set(df.loc[split == s, "patient_id"]) for s in ("train", "val", "test")}
        assert not groups["train"] & groups["test"]
        assert not groups["train"] & groups["val"]
        assert not groups["val"] & groups["test"]
        assert set(split) == {"train", "val", "test"}
    # разные seed → разные разбиения
    assert not (patient_split(df, 0) == patient_split(df, 1)).all()


def test_tabular_encoder_fits_on_train_only(synthetic):
    df = load_pad_metadata(synthetic / "pad_ufes_20", verbose=False)
    split = patient_split(df, seed=0)
    enc = TabularEncoder().fit(df[split == "train"])
    X = enc.transform(df[split == "test"])
    assert X.shape == (int((split == "test").sum()), enc.dim)
    assert np.isfinite(X).all()
    # неизвестная категория и пропуски не ломают кодирование
    one = enc.transform_one({"age": None, "region": "MOON", "bleed": "yes"})
    assert one.shape == (enc.dim,)
    restored = TabularEncoder.from_dict(enc.to_dict())
    assert np.allclose(restored.transform(df.head(5)), enc.transform(df.head(5)))


# ---------------------------------------------------------------- метрики
def test_metrics_basic():
    y = np.array([0, 0, 1, 1])
    assert safe_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert abs(pauc_above_tpr(y, [0.1, 0.2, 0.8, 0.9]) - 0.2) < 1e-9   # идеальная модель → максимум 0,2
    r = binary_report(y, np.array([0.1, 0.6, 0.8, 0.9]), 0.5)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 0)


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 500), 0, 1)
    t = threshold_for_sensitivity(y, p, 0.9)
    assert binary_report(y, p, t)["sensitivity"] >= 0.9


# ---------------------------------------------------------------- модели
@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet18", "vit_tiny_patch16_224"])
def test_models_and_gradcam(backbone):
    x = torch.randn(2, 3, 224, 224)
    m = ImageClassifier(backbone, pretrained=False)
    assert m(x).shape == (2,)
    mm = MultimodalClassifier(backbone, tab_dim=10, pretrained=False)
    assert mm(x, torch.randn(2, 10)).shape == (2,)
    cam, prob = grad_cam(m, x[:1])
    assert cam.shape == (224, 224) and 0 <= prob <= 1 and cam.max() <= 1


# ---------------------------------------------------------------- сквозной прогон
def _run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_train_and_infer(synthetic, tmp_path):
    runs = tmp_path / "runs"
    common = [f"data.pad_root={synthetic / 'pad_ufes_20'}", f"output.runs_dir={runs}", "train.epochs=1", *TINY]
    _run(["scripts/baselines_tabular.py", "--seeds", "0", "--set", *common])
    _run(["scripts/train_pad.py", "--config", "configs/pad_mm_effb0.yaml", "--seeds", "0", "--set", *common])
    run_dir = runs / "pad_mm_effb0" / "seed0"
    for f in ("best.pt", "metrics.json", "threshold.json", "tab_encoder.json", "preds_test.csv", "history.csv"):
        assert (run_dir / f).exists(), f
    preds = pd.read_csv(run_dir / "preds_test.csv")
    assert preds["prob"].between(0, 1).all()

    from dermatriage.inference import TriageModel
    tm = TriageModel(run_dir)
    img = next((synthetic / "pad_ufes_20").rglob("*.png"))
    res = tm.predict(img, {"age": 70, "gender": "MALE", "bleed": "yes"})
    assert res.decision_code in {"refer", "refer_flags", "uncertain", "retake", "observe"}
    assert res.cam_overlay is not None


@pytest.mark.slow
def test_run_all_quick(synthetic, tmp_path):
    out = _run(["scripts/run_all.py", "--quick", "--pad", str(synthetic / "pad_ufes_20"),
                "--isic", str(synthetic / "isic2024"), "--runs", str(tmp_path / "runs"),
                "--reports", str(tmp_path / "reports"), "--quality-out", str(tmp_path / "q.json"),
                "--only", "pad_effb0", "pad_mm_effb0_isic", "--set", *TINY])
    assert (tmp_path / "reports/results/results.md").exists()
    assert (tmp_path / "reports/analysis/pad_mm_effb0_isic/analysis.md").exists()
    assert "Готово" in out

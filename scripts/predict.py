"""Оценка одного снимка из командной строки (без веб-интерфейса).

    python scripts/predict.py --model runs/pad_mm_effb0_isic/seed0 --image photo.jpg \
        --age 64 --gender MALE --region BACK --bleed yes --changed yes --out cam.png
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from dermatriage.inference import DISCLAIMER, TriageModel
from dermatriage.pipeline import resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--age", type=float)
    ap.add_argument("--gender", choices=["MALE", "FEMALE"])
    ap.add_argument("--region")
    for f in ("itch", "grew", "hurt", "changed", "bleed", "elevation"):
        ap.add_argument(f"--{f}", choices=["yes", "no", "unk"], default="unk")
    ap.add_argument("--out", help="куда сохранить Grad-CAM (png)")
    ap.add_argument("--no-red-flags", action="store_true")
    args = ap.parse_args()

    tm = TriageModel(resolve(args.model), use_red_flags=not args.no_red_flags)
    answers = {k: getattr(args, k) for k in ("age", "gender", "region", "itch", "grew", "hurt",
                                              "changed", "bleed", "elevation")}
    res = tm.predict(args.image, answers, explain=bool(args.out))
    if args.out and res.cam_overlay is not None:
        res.cam_overlay.save(args.out)
    print(json.dumps({"decision": res.decision, "code": res.decision_code,
                      "probability": round(res.probability, 4), "threshold": round(res.threshold, 4),
                      "tta_std": round(res.tta_std, 4), "reasons": res.reasons,
                      "quality_ok": res.quality["ok"]}, ensure_ascii=False, indent=2))
    print(DISCLAIMER)


if __name__ == "__main__":
    main()

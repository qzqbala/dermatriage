"""Веб-демо DermaTriage KZ для фельдшера / врача общей практики.

    python app/app.py --model runs/pad_mm_effb0_isic/seed0
    → откройте http://127.0.0.1:7860 (с телефона в той же сети: --host 0.0.0.0)

Сценарий: фото образования со смартфона + короткая анкета → уровень риска,
решение (направить / наблюдать / переснять), объяснение Grad-CAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from dermatriage.inference import DISCLAIMER, TriageModel  # noqa: E402

REGION_RU = {
    "FACE": "лицо", "NOSE": "нос", "EAR": "ухо", "LIP": "губа", "SCALP": "волосистая часть головы",
    "NECK": "шея", "CHEST": "грудь", "BACK": "спина", "ABDOMEN": "живот", "ARM": "плечо",
    "FOREARM": "предплечье", "HAND": "кисть", "THIGH": "бедро", "FOOT": "стопа",
}
YES_NO = ["Да", "Нет", "Не знаю"]
SYMPTOMS = [("bleed", "Кровоточит?"), ("changed", "Менялось (цвет, форма, размер)?"),
            ("grew", "Растёт?"), ("itch", "Чешется?"), ("hurt", "Болит?"),
            ("elevation", "Возвышается над кожей?")]
DECISION_STYLE = {
    "refer": "🔴", "refer_flags": "🟠", "uncertain": "🟡", "retake": "⚪", "observe": "🟢",
}


def region_choices(model: TriageModel) -> list[str]:
    codes = list(REGION_RU)
    if model.tab_encoder is not None:
        vocab = model.tab_encoder.stats.get("categorical", {}).get("region", [])
        codes = [c for c in vocab] or codes
    return [f"{REGION_RU.get(c, c.lower())} ({c})" for c in codes] + ["другое / не указано"]


def build_ui(model: TriageModel) -> gr.Blocks:
    def run(image, age, sex, region, *answers):
        if image is None:
            return "Загрузите фото образования.", None, ""
        ans = {"age": age if age else None,
               "gender": {"Мужской": "MALE", "Женский": "FEMALE"}.get(sex),
               "region": region.split("(")[-1].rstrip(")") if region and "(" in region else None}
        for (field, _), a in zip(SYMPTOMS, answers):
            ans[field] = {"Да": "yes", "Нет": "no"}.get(a, "unk")
        res = model.predict(image, ans, explain=True)
        icon = DECISION_STYLE.get(res.decision_code, "")
        lines = [f"## {icon} {res.decision}", "",
                 f"**Оценка риска:** {res.probability:.0%} (порог направления {res.threshold:.0%})", ""]
        lines += [f"- {r}" for r in res.reasons]
        q = res.quality
        lines += ["", f"<small>Качество снимка: резкость {q['sharpness']:.0f}, яркость {q['brightness']:.0f}, "
                      f"размер {q['size'][0]}×{q['size'][1]}. Устойчивость к отражениям: разброс {res.tta_std:.2f}."
                      "</small>"]
        cam_note = ("Подсвечены области, которые **повышают** оценку риска."
                    if res.cam_max > 0 else "Модель не нашла на снимке областей, повышающих риск.")
        return "\n".join(lines), res.cam_overlay, cam_note

    # delete_cache: загруженные фото удаляются из временной папки Gradio каждые 10 минут
    with gr.Blocks(title="DermaTriage KZ", delete_cache=(600, 600)) as demo:
        gr.Markdown("# DermaTriage KZ — триаж кожных образований (уровень 1)\n"
                    "Фото со смартфона + анкета → направить ли пациента к специалисту.\n\n"
                    f"> ⚠️ {DISCLAIMER}")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Фото образования (крупно, при дневном свете, без вспышки)",
                                 sources=["upload", "webcam", "clipboard"], height=320)
                with gr.Row():
                    age = gr.Number(label="Возраст, лет", precision=0, minimum=0, maximum=120)
                    sex = gr.Radio(["Мужской", "Женский", "Не указано"], value="Не указано", label="Пол")
                region = gr.Dropdown(region_choices(model), label="Где находится образование",
                                     value="другое / не указано")
                answers = [gr.Radio(YES_NO, value="Не знаю", label=label) for _, label in SYMPTOMS]
                btn = gr.Button("Оценить", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Markdown()
                cam = gr.Image(label="Grad-CAM: на что смотрела модель", height=320)
                cam_note = gr.Markdown()
        btn.click(run, inputs=[image, age, sex, region, *answers], outputs=[verdict, cam, cam_note])
        gr.Markdown(f"<small>Модель: `{model.cfg.get('experiment', '')}` · "
                    f"{model.cfg['model']['backbone']} · порог {model.threshold:.3f} "
                    f"(подобран на валидации под чувствительность ≥ "
                    f"{model.cfg['eval']['target_sensitivity']:.0%}).</small>")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pad_mm_effb0_isic/seed0", help="папка обученной модели")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="временная публичная ссылка Gradio")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-red-flags", action="store_true", help="отключить правило красных флагов")
    args = ap.parse_args()
    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    model = TriageModel(path, device=args.device, use_red_flags=not args.no_red_flags)
    build_ui(model).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

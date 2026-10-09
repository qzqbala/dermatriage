"""Загрузка YAML-конфигов с наследованием и переопределением из командной строки.

Пример:
    cfg = load_config("configs/pad_multimodal.yaml", overrides=["train.epochs=5", "seed=1"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    # yaml понимает числа, bool, null, списки [a, b]
    return yaml.safe_load(raw)


def _set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Читает конфиг. Ключ `base:` подключает родительский файл (путь относительно текущего)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(path.parent / base)
        cfg = _deep_update(parent, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Переопределение должно иметь вид key=value, получено: {ov}")
        key, raw = ov.split("=", 1)
        _set_by_path(cfg, key.strip(), _parse_value(raw))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

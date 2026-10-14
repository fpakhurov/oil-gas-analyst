#!/usr/bin/env python3
"""Сборка payload навыка Ouroboros (≥ 7.x) и, по желанию, установка в data/skills/external.

    python integrations/ouroboros/build_skill.py                       # → dist/ouroboros_skill/oilgas_analyst/
    python integrations/ouroboros/build_skill.py --install ~/Ouroboros/data
    python integrations/ouroboros/build_skill.py --install ~/Ouroboros/data --seed   # + готовая база и модель
    python integrations/ouroboros/build_skill.py --install ~/Ouroboros/data --seed --with-anthropic-key  # + ask

Payload самодостаточен: SKILL.md + plugin.py + копия пакета oilgas_analyst. Сторонние
зависимости объявлены в SKILL.md и ставятся Ouroboros в изолированное окружение навыка
после ревью. После установки: Skills → oilgas_analyst → Review → Enable (или попросите
Ouroboros: «проведи ревью и включи навык oilgas_analyst»), затем вызовите refresh_reports.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SKILL_NAME = "oilgas_analyst"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def build(out_dir: Path, with_anthropic_key: bool = False) -> Path:
    target = out_dir / SKILL_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    manifest = (HERE / "skill" / "SKILL.md").read_text(encoding="utf-8")
    if with_anthropic_key:
        manifest = manifest.replace("env_from_settings: [TAVILY_API_KEY]",
                                    "env_from_settings: [ANTHROPIC_API_KEY, TAVILY_API_KEY]", 1)
    (target / "SKILL.md").write_text(manifest, encoding="utf-8")
    shutil.copy2(HERE / "skill" / "plugin.py", target / "plugin.py")
    shutil.copytree(ROOT / "oilgas_analyst", target / "oilgas_analyst", ignore=IGNORE)
    return target


def seed_state(data_dir: Path) -> Path:
    """Засевает state-директорию навыка отчётами, индексом, снимком цен и моделью эмбеддингов из репозитория,
    чтобы навык работал сразу, без refresh_reports и загрузки модели (~470 МБ) с Hugging Face."""
    state = data_dir / "state" / "skills" / SKILL_NAME
    for sub in ("reports", "prices", "chroma"):
        src = ROOT / "data" / sub
        if src.exists():
            shutil.copytree(src, state / "data" / sub, dirs_exist_ok=True, ignore=IGNORE)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for model_dir in hub.glob("models--intfloat--multilingual-e5-*"):
        shutil.copytree(model_dir, state / "hf_cache" / "hub" / model_dir.name, dirs_exist_ok=True, symlinks=True)
    return state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "ouroboros_skill")
    p.add_argument("--install", type=Path, metavar="OUROBOROS_DATA_DIR",
                   help="каталог данных Ouroboros (обычно ~/Ouroboros/data)")
    p.add_argument("--seed", action="store_true",
                   help="вместе с --install: скопировать готовую базу отчётов, индекс и модель в state навыка")
    p.add_argument("--with-anthropic-key", action="store_true",
                   help="запрашивать ANTHROPIC_API_KEY (включает инструмент ask; без выданного ключа навык не включится)")
    args = p.parse_args()

    payload = build(args.out, args.with_anthropic_key)
    files = sorted(str(f.relative_to(payload)) for f in payload.rglob("*") if f.is_file())
    print(f"payload: {payload} ({len(files)} файлов)")
    if args.install:
        dest = args.install.expanduser() / "skills" / "external" / SKILL_NAME
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(payload, dest)
        print(f"установлено: {dest}")
        if args.seed:
            print(f"state засеян: {seed_state(args.install.expanduser())}")
        print(f"Дальше: Skills → {SKILL_NAME} → Review → Enable"
              + ("" if args.seed else ", затем refresh_reports") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())

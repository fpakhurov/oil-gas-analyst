#!/usr/bin/env python3
"""Установка «Нефтегазового аналитика» в форк Ouroboros.

    python integrations/ouroboros/legacy_v6/install.py /path/to/ouroboros [--drive-root /content/drive/MyDrive/Ouroboros]
        [--pip-spec "oilgas-analyst @ git+https://github.com/<you>/oil-gas-analyst"] [--no-core]

Что делает:
1. Копирует плагин в ouroboros/tools/oilgas.py (автообнаружение реестром).
2. Делает oilgas_ask core-инструментом (виден LLM без enable_tools) — если не указан --no-core.
3. Регистрирует новые инструменты в tests/test_smoke.py (EXPECTED_TOOLS), иначе смоук-тест
   Ouroboros (точное совпадение набора инструментов) упадёт на первом же коммите.
4. Добавляет пакет в requirements.txt Ouroboros.
5. Кладёт роль аналитика в базу знаний Ouroboros (memory/knowledge/oil-gas-analyst.md на Drive),
   если указан --drive-root.

Скрипт идемпотентен: повторный запуск ничего не дублирует.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL_NAMES = ["oilgas_ask", "oilgas_search_reports", "oilgas_web_search", "oilgas_forecast",
              "oilgas_supply_shock", "oilgas_add_report", "oilgas_reindex"]


def patch_core_tools(registry: Path) -> bool:
    text = registry.read_text(encoding="utf-8")
    if '"oilgas_ask"' in text:
        return False
    new, n = re.subn(r"(CORE_TOOL_NAMES\s*=\s*\{)", r'\1\n    "oilgas_ask",', text, count=1)
    if n != 1:
        raise SystemExit(f"Не нашёл CORE_TOOL_NAMES в {registry}")
    registry.write_text(new, encoding="utf-8")
    return True


def patch_smoke_test(smoke: Path) -> bool:
    if not smoke.exists():
        return False
    text = smoke.read_text(encoding="utf-8")
    missing = [n for n in TOOL_NAMES if f'"{n}"' not in text]
    if not missing:
        return False
    insertion = "".join(f'    "{n}",\n' for n in missing)
    new, n = re.subn(r"(EXPECTED_TOOLS\s*=\s*\[\n)", lambda m: m.group(1) + insertion, text, count=1)
    if n != 1:
        raise SystemExit(f"Не нашёл EXPECTED_TOOLS в {smoke}")
    smoke.write_text(new, encoding="utf-8")
    return True


def patch_requirements(req: Path, spec: str) -> bool:
    lines = req.read_text(encoding="utf-8").splitlines() if req.exists() else []
    if any("oilgas-analyst" in line or "oil-gas-analyst" in line for line in lines):
        return False
    lines.append(spec)
    req.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ouroboros_repo", type=Path)
    p.add_argument("--drive-root", type=Path, help="DRIVE_ROOT Ouroboros (для базы знаний)")
    p.add_argument("--pip-spec", default=f"-e {HERE.parents[2]}",
                   help="строка для requirements.txt (по умолчанию — editable-путь к этому репозиторию)")
    p.add_argument("--no-core", action="store_true", help="не делать oilgas_ask core-инструментом")
    args = p.parse_args()

    repo = args.ouroboros_repo.resolve()
    tools_dir = repo / "ouroboros" / "tools"
    if not (tools_dir / "registry.py").exists():
        print(f"{repo} не похож на репозиторий Ouroboros (нет ouroboros/tools/registry.py)", file=sys.stderr)
        return 1

    shutil.copy2(HERE / "oilgas.py", tools_dir / "oilgas.py")
    print(f"✓ плагин → {tools_dir / 'oilgas.py'}")
    if not args.no_core and patch_core_tools(tools_dir / "registry.py"):
        print("✓ oilgas_ask добавлен в CORE_TOOL_NAMES")
    if patch_smoke_test(repo / "tests" / "test_smoke.py"):
        print("✓ tests/test_smoke.py: инструменты добавлены в EXPECTED_TOOLS")
    if patch_requirements(repo / "requirements.txt", args.pip_spec):
        print(f"✓ requirements.txt += {args.pip_spec}")
    if args.drive_root:
        kdir = args.drive_root / "memory" / "knowledge"
        kdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "knowledge_oil-gas-analyst.md", kdir / "oil-gas-analyst.md")
        print(f"✓ роль аналитика → {kdir / 'oil-gas-analyst.md'}")
    print("\nДальше: закоммитьте изменения в ветку Ouroboros, задайте ANTHROPIC_API_KEY (и TAVILY_API_KEY) "
          "в секретах Colab и перезапустите (/restart).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

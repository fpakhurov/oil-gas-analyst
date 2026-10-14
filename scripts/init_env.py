#!/usr/bin/env python3
"""Готовит .env для локального запуска: копирует .env.example и генерирует секреты.

    python scripts/init_env.py            # заполнить пустые секреты, ничего не перезаписывая
    python scripts/init_env.py --no-tracing  # без Langfuse (поднимется только аналитик)

Заполняются только пустые значения — ключи API и уже заданные секреты не трогаются.
После запуска впишите ANTHROPIC_API_KEY (и по желанию TAVILY_API_KEY) и выполните
`docker compose up --build`.
"""

from __future__ import annotations

import argparse
import re
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENERATED = {
    "LANGFUSE_PUBLIC_KEY": lambda: "pk-lf-" + secrets.token_hex(16),
    "LANGFUSE_SECRET_KEY": lambda: "sk-lf-" + secrets.token_hex(16),
    "LANGFUSE_INIT_USER_PASSWORD": lambda: secrets.token_urlsafe(12),
    "LANGFUSE_NEXTAUTH_SECRET": lambda: secrets.token_urlsafe(32),
    "LANGFUSE_SALT": lambda: secrets.token_urlsafe(32),
    "LANGFUSE_ENCRYPTION_KEY": lambda: secrets.token_hex(32),  # ровно 64 hex-символа
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(18),
    "CLICKHOUSE_PASSWORD": lambda: secrets.token_urlsafe(18),
    "REDIS_AUTH": lambda: secrets.token_urlsafe(18),
    "MINIO_ROOT_PASSWORD": lambda: secrets.token_urlsafe(18),
}


def set_value(text: str, key: str, value: str, only_if_empty: bool = True) -> tuple[str, bool]:
    pattern = re.compile(rf"^{key}=(.*)$", re.M)
    m = pattern.search(text)
    if m:
        if only_if_empty and m.group(1).strip():
            return text, False
        return pattern.sub(f"{key}={value}", text, count=1), True
    return text.rstrip("\n") + f"\n{key}={value}\n", True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-tracing", action="store_true", help="не включать Langfuse")
    args = p.parse_args()

    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists():
        shutil.copy(example, env)
        print("создан .env из .env.example")
    text = env.read_text(encoding="utf-8")
    changed = []
    for key, gen in GENERATED.items():
        text, did = set_value(text, key, gen())
        if did:
            changed.append(key)
    tracing = "false" if args.no_tracing else "true"
    text, _ = set_value(text, "LANGFUSE_ENABLED", tracing, only_if_empty=False)
    text, _ = set_value(text, "COMPOSE_PROFILES", "" if args.no_tracing else "observability", only_if_empty=False)
    env.write_text(text, encoding="utf-8")
    env.chmod(0o600)

    print(f"сгенерированы: {', '.join(changed) if changed else 'ничего (всё уже задано)'}")
    if not args.no_tracing:
        email = re.search(r"^LANGFUSE_INIT_USER_EMAIL=(.*)$", text, re.M)
        print(f"Langfuse: http://localhost:3000 — логин {email.group(1) if email else 'admin@local.test'}, "
              "пароль — LANGFUSE_INIT_USER_PASSWORD в .env")
    if not re.search(r"^ANTHROPIC_API_KEY=sk-ant-(?!\.\.\.)\S+", text, re.M):
        print("⚠️  впишите ANTHROPIC_API_KEY в .env")
    print("дальше: docker compose up --build")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SQLite-хранилище приложения: список диалогов, отрисованные ходы, учёт расходов, метаданные.

Состояние агента (сообщения, thinking-блоки) хранит LangGraph-checkpointer в отдельной базе;
здесь — то, что нужно интерфейсу и бюджету: заголовки диалогов, готовые к показу ходы
(ответ + доказательства + проверка), расходы по каждому вызову модели.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_thread ON turns(thread_id, id);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    thread_id TEXT,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read INTEGER NOT NULL,
    cache_write INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit выполняет скрипт в разных потоках — одно соединение под общим замком
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

    # --- диалоги ---------------------------------------------------------

    def upsert_thread(self, thread_id: str, title: str, model: str) -> None:
        now = _now()
        self._execute(
            "INSERT INTO threads(thread_id, title, model, created_at, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at",
            (thread_id, title[:120], model, now, now),
        )

    def list_threads(self, limit: int = 30) -> list[dict]:
        rows = self._execute("SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def get_thread(self, thread_id: str) -> dict | None:
        rows = self._execute("SELECT * FROM threads WHERE thread_id=?", (thread_id,))
        return dict(rows[0]) if rows else None

    def delete_thread(self, thread_id: str) -> None:
        self._execute("DELETE FROM turns WHERE thread_id=?", (thread_id,))
        self._execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))

    def add_turn(self, thread_id: str, question: str, payload: dict) -> None:
        self._execute(
            "INSERT INTO turns(thread_id, created_at, question, payload) VALUES (?,?,?,?)",
            (thread_id, _now(), question, json.dumps(payload, ensure_ascii=False, default=str)),
        )

    def get_turns(self, thread_id: str) -> list[dict]:
        rows = self._execute("SELECT question, payload FROM turns WHERE thread_id=? ORDER BY id", (thread_id,))
        return [{"question": r["question"], **json.loads(r["payload"])} for r in rows]

    # --- расходы ---------------------------------------------------------

    def record_usage(self, thread_id: str | None, model: str, input_tokens: int, output_tokens: int,
                     cache_read: int, cache_write: int, cost_usd: float) -> None:
        self._execute(
            "INSERT INTO usage(ts, thread_id, model, input_tokens, output_tokens, cache_read, cache_write, cost_usd) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_now(), thread_id, model, input_tokens, output_tokens, cache_read, cache_write, cost_usd),
        )

    def spent_today(self) -> float:
        """Расходы за текущие сутки (UTC), $."""
        day = datetime.now(timezone.utc).date().isoformat()
        rows = self._execute("SELECT COALESCE(SUM(cost_usd), 0) AS s FROM usage WHERE ts >= ?", (day,))
        return float(rows[0]["s"])

    # --- метаданные ------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        rows = self._execute("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._execute("INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))


@lru_cache
def _store(path: str) -> AppStore:
    return AppStore(Path(path))


def get_store(settings) -> AppStore:
    return _store(str(settings.app_db))

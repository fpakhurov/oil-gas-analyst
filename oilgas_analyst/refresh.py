"""Автообновление базы знаний: новые выпуски EIA STEO, переиндексация, снимок цен.

Запускается из UI фоновым потоком (раз в час проверяет, не пора ли) или вручную:
`oilgas refresh`. Работает в том же процессе, что и поиск, — один клиент Chroma,
без конкурентной записи из нескольких процессов.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.storage import get_store

log = logging.getLogger(__name__)

META_KEY = "kb_refresh"
_lock = threading.Lock()


def last_refresh(settings: Settings | None = None) -> dict | None:
    raw = get_store(settings or get_settings()).get_meta(META_KEY)
    return json.loads(raw) if raw else None


def refresh_knowledge_base(settings: Settings | None = None, months: int = 2) -> dict:
    from oilgas_analyst.forecasting.data import save_snapshot
    from oilgas_analyst.rag.download import download_reports
    from oilgas_analyst.rag.store import ingest_reports

    s = settings or get_settings()
    with _lock:
        result: dict = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            result["download"] = download_reports(months=months, settings=s)
            result["ingest"] = ingest_reports(settings=s)
        except Exception as e:
            log.warning("Обновление отчётов не удалось: %s", e)
            result["error"] = f"{type(e).__name__}: {e}"
        try:
            result["prices"] = save_snapshot(s)
        except Exception as e:  # снимок цен — офлайн-резерв, его отсутствие не критично
            result["prices_error"] = f"{type(e).__name__}: {e}"
        get_store(s).set_meta(META_KEY, json.dumps(result, ensure_ascii=False))
        return result


def is_due(settings: Settings | None = None, now: datetime | None = None) -> bool:
    s = settings or get_settings()
    if s.refresh_interval_days <= 0:
        return False
    last = last_refresh(s)
    if not last:
        return True
    at = datetime.fromisoformat(last["at"])
    return (now or datetime.now(timezone.utc)) - at >= timedelta(days=s.refresh_interval_days)


def start_scheduler(settings: Settings | None = None, check_every_s: int = 3600) -> threading.Thread | None:
    """Фоновый поток: раз в час проверяет, не пора ли обновить базу. Возвращает поток или None."""
    s = settings or get_settings()
    if s.refresh_interval_days <= 0:
        return None
    stop = threading.Event()

    def loop():
        stop.wait(30)  # не мешаем старту приложения
        while not stop.is_set():
            try:
                if is_due(s):
                    log.info("Плановое обновление базы отчётов")
                    refresh_knowledge_base(s)
            except Exception:
                log.exception("Плановое обновление базы упало")
            stop.wait(check_every_s)

    t = threading.Thread(target=loop, name="kb-refresh", daemon=True)
    t.stop_event = stop  # type: ignore[attr-defined]
    t.start()
    return t

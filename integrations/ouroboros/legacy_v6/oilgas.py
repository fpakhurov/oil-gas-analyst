"""Ouroboros tool plugin: «Нефтегазовый аналитик».

Этот модуль копируется в `ouroboros/tools/oilgas.py` форка Ouroboros
(см. integrations/ouroboros/install.py). Реестр Ouroboros автоматически находит
модули с функцией get_tools() — дополнительных регистраций не нужно.

Ядро аналитика (LangGraph-агент, RAG, веб-поиск, расчётный модуль) живёт в пакете
`oilgas_analyst`; здесь — тонкий адаптер в контракт Ouroboros:
handler(ctx: ToolContext, **args) -> str.

Инструменты:
  oilgas_ask             — полный цикл аналитика (RAG → веб → расчёты) с маркировкой источников;
  oilgas_search_reports  — поиск по базе отчётов (OPEC/EIA/IEA);
  oilgas_web_search      — веб-поиск с фильтром жёлтой прессы;
  oilgas_forecast        — прогноз Brent/WTI (SARIMA + ETS) с доверительным интервалом;
  oilgas_supply_shock    — сценарий изменения добычи на N млн б/с;
  oilgas_add_report      — скачать PDF-отчёт по ссылке, добавить в каталог и проиндексировать;
  oilgas_reindex         — переиндексировать data/reports.

Данные (отчёты, Chroma, кэш цен) хранятся в $OILGAS_DATA_DIR — по умолчанию
<drive_root>/oilgas, чтобы база переживала перезапуски Colab.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import List

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

_lock = threading.Lock()
_analyst = None
_threads: dict[str, str] = {}


def _ensure_env(ctx: ToolContext) -> None:
    """Направляет данные аналитика на Drive Ouroboros (если DATA_DIR не задан явно)."""
    if "DATA_DIR" not in os.environ:
        os.environ["DATA_DIR"] = os.environ.get("OILGAS_DATA_DIR") or str(Path(ctx.drive_root) / "oilgas")


def _settings(ctx: ToolContext):
    _ensure_env(ctx)
    try:
        from oilgas_analyst.config import get_settings
    except ImportError as e:  # пакет не установлен в окружение Ouroboros
        raise RuntimeError(
            "Пакет oilgas_analyst не установлен. Выполните: pip install 'oilgas-analyst @ git+<repo-url>' "
            "или pip install -e <путь к oil-gas-analyst>"
        ) from e
    return get_settings()


def _get_analyst(ctx: ToolContext):
    global _analyst
    with _lock:
        if _analyst is None:
            from oilgas_analyst.agent import OilGasAnalyst

            _analyst = OilGasAnalyst(_settings(ctx))
        return _analyst


def _err(name: str, e: Exception) -> str:
    log.warning("%s failed", name, exc_info=True)
    return f"⚠️ {name}: {type(e).__name__}: {e}"


# --- handlers ---------------------------------------------------------------

def _ask(ctx: ToolContext, question: str, conversation: str = "default") -> str:
    try:
        analyst = _get_analyst(ctx)
        # Отдельный поток диалога на чат Ouroboros + метку беседы: аналитик помнит контекст
        key = f"{ctx.current_chat_id}:{conversation}"
        thread_id = _threads.setdefault(key, f"ouroboros-{len(_threads)}-{conversation}")
        turn = analyst.ask(question, thread_id)
        trace = ", ".join(t["name"] for t in turn.tool_calls) or "только база отчётов"
        return f"{turn.answer}\n\n[trace: автопоиск по отчётам → {trace}]"
    except Exception as e:
        return _err("oilgas_ask", e)


def _search_reports(ctx: ToolContext, query: str, publisher: str = "", k: int = 6) -> str:
    try:
        s = _settings(ctx)
        from oilgas_analyst.rag.store import format_chunks, search_reports

        return format_chunks(search_reports(query, k=k, publisher=publisher or None, settings=s),
                             s.rag_relevance_threshold)
    except Exception as e:
        return _err("oilgas_search_reports", e)


def _web_search(ctx: ToolContext, query: str, topic: str = "news", days: int = 30) -> str:
    try:
        s = _settings(ctx)
        from oilgas_analyst.web.search import format_results, web_search

        results, stats = web_search(query, topic=topic, days=days, settings=s)
        return format_results(results, stats)
    except Exception as e:
        return _err("oilgas_web_search", e)


def _forecast(ctx: ToolContext, benchmark: str = "brent", horizon_months: int = 3,
              confidence: float = 0.8, method: str = "ensemble") -> str:
    try:
        s = _settings(ctx)
        from oilgas_analyst.forecasting.service import run_forecast

        return run_forecast(benchmark, horizon_months, confidence, method, s)[0]
    except Exception as e:
        return _err("oilgas_forecast", e)


def _supply_shock(ctx: ToolContext, supply_change_mbd: float, benchmark: str = "brent",
                  horizon_months: int = 6, offset_share: float = 0.0, phase_in_months: int = 2) -> str:
    try:
        s = _settings(ctx)
        from oilgas_analyst.forecasting.service import run_supply_shock

        return run_supply_shock(supply_change_mbd, benchmark, horizon_months, offset_share,
                                phase_in_months, settings=s)[0]
    except Exception as e:
        return _err("oilgas_supply_shock", e)


def _add_report(ctx: ToolContext, url: str, publisher: str, title: str, short_name: str, date: str) -> str:
    try:
        s = _settings(ctx)
        import requests

        from oilgas_analyst.rag.catalog import ReportMeta, load_catalog, save_catalog
        from oilgas_analyst.rag.store import ingest_reports

        if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", date):
            return "⚠️ date должен быть в формате YYYY-MM или YYYY-MM-DD"
        slug = re.sub(r"[^a-z0-9]+", "_", short_name.lower()).strip("_")
        fname = f"{slug}_{date}.pdf"
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (ouroboros oilgas)"})
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            return (f"⚠️ По ссылке не PDF (HTTP {resp.status_code}). Если сайт за Cloudflare — скачай файл "
                    f"через browse_page и положи в {s.reports_dir}, затем вызови oilgas_reindex.")
        s.reports_dir.mkdir(parents=True, exist_ok=True)
        (s.reports_dir / fname).write_bytes(resp.content)
        catalog = load_catalog(s.reports_dir)
        catalog[fname] = ReportMeta(fname, publisher, title, short_name, date, url)
        save_catalog(s.reports_dir, catalog)
        stats = ingest_reports(settings=s)
        return f"✅ {fname} добавлен: {json.dumps(stats, ensure_ascii=False)}"
    except Exception as e:
        return _err("oilgas_add_report", e)


def _reindex(ctx: ToolContext, force: bool = False) -> str:
    try:
        s = _settings(ctx)
        from oilgas_analyst.rag.store import ingest_reports

        return json.dumps(ingest_reports(force=force, settings=s), ensure_ascii=False, indent=2)
    except Exception as e:
        return _err("oilgas_reindex", e)


# --- registry ---------------------------------------------------------------

def _schema(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


_BENCH = {"type": "string", "enum": ["brent", "wti"], "description": "Бенчмарк (Urals не поддерживается)"}


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("oilgas_ask", _schema(
            "oilgas_ask",
            "Нефтегазовый аналитик: полный ответ на отраслевой вопрос (Upstream/Midstream/Downstream, "
            "Brent/WTI/Urals, ОПЕК+, санкции, баланс рынка). Сначала база отчётов OPEC/EIA/IEA, затем веб, "
            "при необходимости — прогноз цены. Ответ с маркировкой источников.",
            {"question": {"type": "string"},
             "conversation": {"type": "string", "description": "Метка беседы для сохранения контекста"}},
            ["question"]), _ask, timeout_sec=600),
        ToolEntry("oilgas_search_reports", _schema(
            "oilgas_search_reports",
            "Поиск по базе отраслевых отчётов (OPEC MOMR, EIA STEO, IEA OMR); фрагменты с метками "
            "[Отчёт …, месяц год, стр. N]. Запрос лучше на английском.",
            {"query": {"type": "string"}, "publisher": {"type": "string", "enum": ["", "OPEC", "EIA", "IEA"]},
             "k": {"type": "integer"}}, ["query"]), _search_reports, timeout_sec=180),
        ToolEntry("oilgas_web_search", _schema(
            "oilgas_web_search",
            "Веб-поиск по нефтегазовой тематике (Tavily/DuckDuckGo) с отсевом жёлтой прессы и соцсетей "
            "и ранжированием по надёжности источника.",
            {"query": {"type": "string"}, "topic": {"type": "string", "enum": ["news", "general"]},
             "days": {"type": "integer"}}, ["query"]), _web_search, timeout_sec=120),
        ToolEntry("oilgas_forecast", _schema(
            "oilgas_forecast",
            "Прогноз среднемесячной цены Brent/WTI (SARIMA + ETS, ансамбль по бэктесту) с доверительным "
            "интервалом и интерпретацией.",
            {"benchmark": _BENCH, "horizon_months": {"type": "integer", "minimum": 1, "maximum": 24},
             "confidence": {"type": "number"}, "method": {"type": "string", "enum": ["ensemble", "sarima", "ets"]}},
            []), _forecast, timeout_sec=300),
        ToolEntry("oilgas_supply_shock", _schema(
            "oilgas_supply_shock",
            "Сценарий: диапазон цены при изменении мирового предложения на N млн б/с "
            "(отрицательное значение — сокращение добычи).",
            {"supply_change_mbd": {"type": "number"}, "benchmark": _BENCH,
             "horizon_months": {"type": "integer"}, "offset_share": {"type": "number"},
             "phase_in_months": {"type": "integer"}},
            ["supply_change_mbd"]), _supply_shock, timeout_sec=300),
        ToolEntry("oilgas_add_report", _schema(
            "oilgas_add_report",
            "Добавить новый PDF-отчёт в базу аналитика по прямой ссылке и проиндексировать его.",
            {"url": {"type": "string"}, "publisher": {"type": "string", "description": "OPEC / EIA / IEA / …"},
             "title": {"type": "string"}, "short_name": {"type": "string", "description": "напр. 'OPEC MOMR'"},
             "date": {"type": "string", "description": "YYYY-MM"}},
            ["url", "publisher", "title", "short_name", "date"]), _add_report, timeout_sec=900),
        ToolEntry("oilgas_reindex", _schema(
            "oilgas_reindex",
            "Переиндексировать PDF-отчёты в $OILGAS_DATA_DIR/reports (после ручного добавления файлов).",
            {"force": {"type": "boolean"}}, []), _reindex, timeout_sec=1800),
    ]

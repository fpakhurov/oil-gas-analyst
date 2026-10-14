"""oilgas_analyst — Ouroboros extension: «Нефтегазовый аналитик».

Тонкий адаптер PluginAPI 2.0 над пакетом ``oilgas_analyst`` (лежит рядом в payload).
Зависимости навыка (LangGraph, Chroma, sentence-transformers, statsmodels…) ставятся
в изолированное окружение навыка, поэтому хост вызывает обработчики в короткоживущем
дочернем процессе: тяжёлые импорты делаем лениво, внутри обработчиков.

Все данные — в state-директории навыка: отчёты, индекс Chroma, снимок цен, кэш модели.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_api: Any = None
_SETTINGS_KEYS = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
_MAX_QUESTION = 4000


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------


def _state_dir() -> pathlib.Path:
    if _api is None:
        raise RuntimeError("extension is not registered")
    getter = getattr(_api, "get_state_dir", None)
    path = pathlib.Path(getter() if getter else _api.get_runtime_info()["state_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure() -> None:
    """Направляет данные и ключи пакета в state-директорию и grant-нутые настройки."""
    state = _state_dir()
    os.environ["DATA_DIR"] = str(state / "data")
    os.environ.setdefault("HF_HOME", str(state / "hf_cache"))
    # Модель уже в кэше (засеяна установщиком или прошлым вызовом) — не ходим в сеть за проверкой версий
    if any((pathlib.Path(os.environ["HF_HOME"]) / "hub").glob("models--intfloat--multilingual-e5-*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Ключи запрашиваем по одному: не объявленный в манифесте (сборка для подписки) или не выданный
    # владельцем ключ не должен мешать остальным
    for key in _SETTINGS_KEYS:
        try:
            value = (_api.get_settings([key]) or {}).get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)
    from oilgas_analyst.config import get_settings

    get_settings.cache_clear()


def _settings():
    _configure()
    from oilgas_analyst.config import get_settings

    return get_settings()


def _error(name: str, exc: Exception) -> str:
    if _api is not None:
        _api.log("warning", f"oilgas_analyst.{name}: {type(exc).__name__}: {exc}")
    return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструменты (kwargs-only, возвращают строку)
# ---------------------------------------------------------------------------


def _tool_ask(*, question: str = "") -> str:
    question = (question or "").strip()
    if not question:
        return json.dumps({"error": "question is empty"}, ensure_ascii=False)
    if len(question) > _MAX_QUESTION:
        return json.dumps({"error": f"question too long (max {_MAX_QUESTION} chars)"}, ensure_ascii=False)
    try:
        s = _settings()
        if not s.anthropic_api_key:
            return json.dumps({"error": (
                "ANTHROPIC_API_KEY не выдан навыку, встроенный LangGraph-аналитик недоступен. Ответь сам в роли "
                "аналитика по правилам навыка: search_reports (запрос по-английски) → при необходимости "
                "web_search → forecast / supply_shock; каждую цифру помечай меткой из результатов инструментов."
            )}, ensure_ascii=False)
        from oilgas_analyst.agent import OilGasAnalyst

        turn = OilGasAnalyst(s).ask(question)
        trace = ", ".join(t["name"] for t in turn.tool_calls) or "только база отчётов"
        return f"{turn.answer}\n\n[trace: автопоиск по отчётам → {trace}]"
    except Exception as exc:
        return _error("ask", exc)


def _tool_search_reports(*, query: str = "", publisher: str = "", k: int = 6) -> str:
    try:
        s = _settings()
        from oilgas_analyst.rag.store import format_chunks, search_reports

        k = min(max(int(k or 6), 1), 20)
        chunks = search_reports(query, k=k, publisher=publisher or None, settings=s)
        return format_chunks(chunks, s.rag_relevance_threshold)
    except Exception as exc:
        return _error("search_reports", exc)


def _tool_web_search(*, query: str = "", topic: str = "news", days: int = 30) -> str:
    try:
        s = _settings()
        from oilgas_analyst.web.search import format_results, web_search

        topic = topic if topic in ("news", "general") else "news"
        results, stats = web_search(query, topic=topic, days=int(days or 30), settings=s)
        return format_results(results, stats)
    except Exception as exc:
        return _error("web_search", exc)


def _tool_forecast(*, benchmark: str = "brent", horizon_months: int = 3, confidence: float = 0.8,
                   method: str = "ensemble") -> str:
    try:
        s = _settings()
        from oilgas_analyst.forecasting.service import run_forecast

        return run_forecast(benchmark, int(horizon_months), float(confidence), method, s)[0]
    except Exception as exc:
        return _error("forecast", exc)


def _tool_supply_shock(*, supply_change_mbd: float = 0.0, benchmark: str = "brent", horizon_months: int = 6,
                       offset_share: float = 0.0, phase_in_months: int = 2) -> str:
    try:
        s = _settings()
        from oilgas_analyst.forecasting.service import run_supply_shock

        return run_supply_shock(float(supply_change_mbd), benchmark, int(horizon_months), float(offset_share),
                                int(phase_in_months), settings=s)[0]
    except Exception as exc:
        return _error("supply_shock", exc)


def _tool_refresh_reports(*, months: int = 2) -> str:
    try:
        s = _settings()
        from oilgas_analyst.forecasting.data import save_snapshot
        from oilgas_analyst.rag.download import download_reports
        from oilgas_analyst.rag.store import ingest_reports

        months = min(max(int(months or 2), 1), 6)
        result = {"download": download_reports(months=months, settings=s)}
        result["ingest"] = ingest_reports(settings=s)
        try:
            result["prices"] = save_snapshot(s)
        except Exception as exc:  # снимок цен — необязательный офлайн-резерв
            result["prices"] = f"skipped: {exc}"
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return _error("refresh_reports", exc)


def _tool_add_report(*, url: str = "", publisher: str = "", title: str = "", short_name: str = "",
                     date: str = "") -> str:
    if not re.fullmatch(r"https?://\S+", url or ""):
        return json.dumps({"error": "url must be an http(s) link to a PDF"}, ensure_ascii=False)
    if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", date or ""):
        return json.dumps({"error": "date must be YYYY-MM or YYYY-MM-DD"}, ensure_ascii=False)
    if not short_name.strip():
        return json.dumps({"error": "short_name is required, e.g. 'OPEC MOMR'"}, ensure_ascii=False)
    try:
        s = _settings()
        import requests

        from oilgas_analyst.rag.catalog import ReportMeta, load_catalog, save_catalog
        from oilgas_analyst.rag.store import ingest_reports

        slug = re.sub(r"[^a-z0-9]+", "_", short_name.lower()).strip("_")[:40]
        fname = f"{slug}_{date}.pdf"
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (Ouroboros oilgas_analyst)"})
        content = resp.content[: 80 * 1024 * 1024]
        if resp.status_code != 200 or not content.startswith(b"%PDF"):
            return json.dumps({"error": f"not a PDF (HTTP {resp.status_code}); if the site is behind "
                                        "Cloudflare, find another direct link"}, ensure_ascii=False)
        s.reports_dir.mkdir(parents=True, exist_ok=True)
        (s.reports_dir / fname).write_bytes(content)
        catalog = load_catalog(s.reports_dir)
        catalog[fname] = ReportMeta(fname, publisher or "Unknown", title or short_name, short_name, date, url)
        save_catalog(s.reports_dir, catalog)
        return json.dumps({"added": fname, "ingest": ingest_reports(settings=s)}, ensure_ascii=False)
    except Exception as exc:
        return _error("add_report", exc)


# ---------------------------------------------------------------------------
# HTTP-маршруты для виджета
# ---------------------------------------------------------------------------


async def _read_json(request: Any) -> Dict[str, Any]:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


async def _route_ask(request: Any):
    from starlette.responses import JSONResponse

    body = await _read_json(request)
    answer = await asyncio.to_thread(_tool_ask, question=str(body.get("question", "")))
    if answer.startswith("{\"error\""):
        return JSONResponse({"answer": f"⚠️ {json.loads(answer)['error']}"}, status_code=200)
    return JSONResponse({"answer": answer})


async def _route_forecast(request: Any):
    from starlette.responses import JSONResponse

    body = await _read_json(request)
    benchmark = str(body.get("benchmark") or "brent").strip().lower()
    if benchmark not in ("brent", "wti"):
        return JSONResponse({"report": "⚠️ Бенчмарк: brent или wti"})
    try:
        horizon = min(max(int(str(body.get("horizon_months") or "3").strip()), 1), 24)
        shock_raw = str(body.get("supply_change_mbd") or "").strip().replace(",", ".")
        shock = float(shock_raw) if shock_raw else None
    except ValueError:
        return JSONResponse({"report": "⚠️ Горизонт — целое число месяцев, шок — число (млн б/с)"})
    if shock is None:
        text = await asyncio.to_thread(_tool_forecast, benchmark=benchmark, horizon_months=horizon)
    else:
        text = await asyncio.to_thread(_tool_supply_shock, supply_change_mbd=shock, benchmark=benchmark,
                                       horizon_months=horizon)
    return JSONResponse({"report": f"```\n{text}\n```"})


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

_BENCH = {"type": "string", "enum": ["brent", "wti"], "description": "Бенчмарк (Urals не поддерживается)"}

_UI_RENDER = {
    "kind": "declarative",
    "schema_version": 1,
    "components": [
        {
            "type": "form",
            "route": "ask",
            "method": "POST",
            "target": "answer",
            "fields": [
                {
                    "name": "question",
                    "label": "Вопрос аналитику",
                    "type": "text",
                    "placeholder": "Например: что ОПЕК+ решила по добыче и как это повлияет на Brent?",
                    "required": True,
                },
            ],
            "submit_label": "Спросить",
        },
        {"type": "markdown", "target": "answer", "path": "answer"},
        {
            "type": "form",
            "route": "forecast",
            "method": "POST",
            "target": "forecast",
            "fields": [
                {"name": "benchmark", "label": "Бенчмарк (brent / wti)", "type": "text", "placeholder": "brent"},
                {"name": "horizon_months", "label": "Горизонт, мес.", "type": "text", "placeholder": "3"},
                {
                    "name": "supply_change_mbd",
                    "label": "Шок предложения, млн б/с (минус — сокращение; пусто — без сценария)",
                    "type": "text",
                    "placeholder": "-1.5",
                },
            ],
            "submit_label": "Рассчитать",
        },
        {"type": "markdown", "target": "forecast", "path": "report"},
    ],
}


def register(api: Any) -> None:
    """PluginAPI 2.0 entry point."""
    global _api
    _api = api

    api.register_tool(
        "ask", _tool_ask,
        description=("Нефтегазовый аналитик (LangGraph + Claude API): полный ответ на отраслевой вопрос "
                     "(Upstream/Midstream/Downstream, Brent/WTI/Urals, ОПЕК+, санкции, баланс рынка) с метками "
                     "источников. Нужен выданный ANTHROPIC_API_KEY; без него отвечай сам через search_reports, "
                     "web_search, forecast и supply_shock."),
        schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        timeout_sec=300,
    )
    api.register_tool(
        "search_reports", _tool_search_reports,
        description="Поиск по базе отчётов OPEC MOMR / EIA STEO / IEA OMR. Фрагменты с метками [Отчёт …, стр. N].",
        schema={"type": "object", "properties": {
            "query": {"type": "string", "description": "Запрос, лучше на английском"},
            "publisher": {"type": "string", "enum": ["", "OPEC", "EIA", "IEA"]},
            "k": {"type": "integer", "description": "Число фрагментов, 1–20"}}, "required": ["query"]},
        timeout_sec=120,
    )
    api.register_tool(
        "web_search", _tool_web_search,
        description="Веб-поиск по нефтегазовой тематике с отсевом жёлтой прессы, соцсетей и кликбейта.",
        schema={"type": "object", "properties": {
            "query": {"type": "string"}, "topic": {"type": "string", "enum": ["news", "general"]},
            "days": {"type": "integer"}}, "required": ["query"]},
        timeout_sec=90,
    )
    api.register_tool(
        "forecast", _tool_forecast,
        description="Прогноз среднемесячной цены Brent/WTI (SARIMA + ETS, ансамбль) с доверительным интервалом.",
        schema={"type": "object", "properties": {
            "benchmark": _BENCH, "horizon_months": {"type": "integer", "minimum": 1, "maximum": 24},
            "confidence": {"type": "number"}, "method": {"type": "string", "enum": ["ensemble", "sarima", "ets"]}},
            "required": []},
        timeout_sec=240,
    )
    api.register_tool(
        "supply_shock", _tool_supply_shock,
        description="Диапазон цены при изменении мирового предложения на N млн б/с (минус — сокращение добычи).",
        schema={"type": "object", "properties": {
            "supply_change_mbd": {"type": "number"}, "benchmark": _BENCH,
            "horizon_months": {"type": "integer"}, "offset_share": {"type": "number"},
            "phase_in_months": {"type": "integer"}}, "required": ["supply_change_mbd"]},
        timeout_sec=240,
    )
    api.register_tool(
        "refresh_reports", _tool_refresh_reports,
        description=("Скачать свежие выпуски EIA STEO, проиндексировать базу отчётов и обновить снимок цен. "
                     "Идемпотентно: при таймауте просто вызови ещё раз."),
        schema={"type": "object", "properties": {"months": {"type": "integer", "minimum": 1, "maximum": 6}},
                "required": []},
        timeout_sec=300,
    )
    api.register_tool(
        "add_report", _tool_add_report,
        description="Добавить PDF-отчёт (например, новый OPEC MOMR) по прямой ссылке и проиндексировать.",
        schema={"type": "object", "properties": {
            "url": {"type": "string"}, "publisher": {"type": "string", "description": "OPEC / EIA / IEA / …"},
            "title": {"type": "string"}, "short_name": {"type": "string", "description": "напр. 'OPEC MOMR'"},
            "date": {"type": "string", "description": "YYYY-MM"}},
            "required": ["url", "short_name", "date"]},
        timeout_sec=300,
    )
    api.register_route("ask", _route_ask, methods=("POST",))
    api.register_route("forecast", _route_forecast, methods=("POST",))
    api.register_ui_tab("analyst", "Нефтегазовый аналитик", icon="🛢", render=_UI_RENDER)

    def _cleanup() -> None:
        global _api
        _api = None

    api.on_unload(_cleanup)
    api.log("info", "oilgas_analyst: extension registered (7 tools, 2 routes)")


__all__ = ["register"]

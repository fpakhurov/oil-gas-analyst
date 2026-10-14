"""LangChain-инструменты агента. Каждый возвращает (текст для LLM, артефакт для UI)."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from oilgas_analyst.config import get_settings
from oilgas_analyst.forecasting.service import run_forecast, run_supply_shock
from oilgas_analyst.rag.store import chunk_sources, format_chunks, search_reports as _search_reports
from oilgas_analyst.web.search import format_results, web_search as _web_search


@tool(response_format="content_and_artifact")
def search_reports(query: str, publisher: Literal["OPEC", "EIA", "IEA"] | None = None, k: int = 6):
    """Поиск по базе отраслевых отчётов (OPEC MOMR, EIA STEO, IEA OMR и др.) — приоритетный источник.

    Args:
        query: поисковый запрос, лучше на английском (отчёты англоязычные), например
            "world oil demand growth forecast 2027".
        publisher: ограничить поиск одним издателем (необязательно).
        k: число фрагментов (по умолчанию 6).
    """
    s = get_settings()
    chunks = _search_reports(query, k=k, publisher=publisher)
    return format_chunks(chunks, s.rag_relevance_threshold), {"kind": "rag", "query": query,
                                                              "sources": chunk_sources(chunks)}


@tool(response_format="content_and_artifact")
def web_search(query: str, topic: Literal["news", "general"] = "news", days: int = 30):
    """Веб-поиск по открытым источникам для актуальных данных: свежие новости, текущие котировки,
    решения ОПЕК+, санкции, заявления регуляторов. Жёлтая пресса и соцсети отфильтровываются автоматически.

    Args:
        query: поисковый запрос (можно на русском или английском).
        topic: "news" — новости (по умолчанию), "general" — общий поиск.
        days: глубина поиска новостей в днях (по умолчанию 30).
    """
    results, stats = _web_search(query, topic=topic, days=days)
    sources = [{"label": r.label(), "name": r.source_name, "url": r.url, "title": r.title, "tier": r.tier_label,
                "published": r.published, "snippet": r.snippet} for r in results]
    return format_results(results, stats), {"kind": "web", "query": query, "sources": sources,
                                             "dropped": stats["dropped"], "provider": stats["provider"]}


@tool(response_format="content_and_artifact")
def forecast_oil_price(benchmark: Literal["brent", "wti"] = "brent", horizon_months: int = 3,
                       confidence: float = 0.8, method: Literal["ensemble", "sarima", "ets"] = "ensemble"):
    """Расчётный модуль: статистический прогноз среднемесячной цены нефти с доверительным интервалом.
    Загружает исторические цены (Yahoo Finance / FRED-EIA / CSV), строит SARIMA и ETS,
    взвешивает их по ошибке бэктеста и возвращает прогноз, интервал и интерпретацию.

    Args:
        benchmark: "brent" или "wti".
        horizon_months: горизонт прогноза в месяцах (1–24).
        confidence: уровень доверительного интервала (например, 0.8 или 0.95).
        method: "ensemble" (по умолчанию), "sarima" или "ets".
    """
    text, payload, _ = run_forecast(benchmark, horizon_months, confidence, method)
    return text, payload


@tool(response_format="content_and_artifact")
def supply_shock_scenario(supply_change_mbd: float, benchmark: Literal["brent", "wti"] = "brent",
                          horizon_months: int = 6, offset_share: float = 0.0, phase_in_months: int = 2,
                          confidence: float = 0.8):
    """Сценарный модуль: оценка диапазона цены при изменении мирового предложения нефти
    (например, сокращение добычи ОПЕК+ на N млн б/с). Базой служит статистический прогноз,
    эффект шока оценивается через краткосрочные эластичности спроса и предложения (3 сценария).

    Args:
        supply_change_mbd: изменение предложения в млн б/с; ОТРИЦАТЕЛЬНОЕ значение — сокращение
            (например, -1.0 для сокращения на 1 млн б/с), положительное — рост добычи.
        benchmark: "brent" или "wti".
        horizon_months: горизонт в месяцах (1–24).
        offset_share: доля шока, компенсируемая другими производителями/запасами (0–1).
        phase_in_months: за сколько месяцев эффект выходит на полную силу.
        confidence: уровень доверительного интервала базового прогноза.
    """
    text, payload, _ = run_supply_shock(supply_change_mbd, benchmark, horizon_months, offset_share,
                                        phase_in_months, confidence)
    return text, payload


ALL_TOOLS = [search_reports, web_search, forecast_oil_price, supply_shock_scenario]

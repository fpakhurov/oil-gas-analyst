"""Веб-поиск: Tavily (если задан ключ) с fallback на DuckDuckGo + фильтр источников."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.web.source_filter import assess, blocked_domains

log = logging.getLogger(__name__)


WEB_TIMEOUT_S = 20


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    published: str
    source_name: str
    tier: int
    tier_label: str

    def label(self) -> str:
        date = f", {self.published[:10]}" if self.published else ""
        return f"[Источник: {self.source_name}, web{date}]"


def _tavily(query: str, max_results: int, topic: str, days: int | None, api_key: str) -> list[dict]:
    from tavily import TavilyClient

    # Таймаут клиента по умолчанию — 60 с; для интерактивного ответа лучше быстро уйти на DuckDuckGo
    kwargs = dict(query=query, max_results=max_results, topic=topic, search_depth="advanced",
                  exclude_domains=blocked_domains()[:150], timeout=WEB_TIMEOUT_S)
    if topic == "news" and days:
        kwargs["days"] = days
    resp = TavilyClient(api_key=api_key).search(**kwargs)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", ""),
         "published": r.get("published_date") or ""}
        for r in resp.get("results", [])
    ]


def _ddg(query: str, max_results: int, topic: str, days: int | None) -> list[dict]:
    from ddgs import DDGS

    timelimit = None
    if days:
        timelimit = "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else "y"
    errors = []
    # Бэкенды ddgs периодически отваливаются по таймауту, а его собственный DNS-резолвер не всегда
    # работает за прокси — поэтому порядок: новости ddgs (с датами) → HTML-версия DDG → текст ddgs
    attempts = (["news"] if topic == "news" else []) + ["html", "text"]
    for kind in attempts:
        try:
            if kind == "html":
                out = _ddg_html(query, max_results, timelimit)
            else:
                with DDGS(timeout=10) as ddgs:
                    if kind == "news":
                        raw = ddgs.news(query, max_results=max_results, timelimit=timelimit)
                        out = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("body", ""),
                                "published": r.get("date", "")} for r in raw]
                    else:
                        raw = ddgs.text(query, max_results=max_results, timelimit=timelimit)
                        out = [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""),
                                "published": ""} for r in raw]
            if out:
                return out
        except Exception as e:
            log.info("DDG %s: %s", kind, e)
            errors.append(f"{kind}: {e}")
    if errors:
        raise RuntimeError("DuckDuckGo недоступен: " + "; ".join(errors[-2:]))
    return []


def iso_date(raw: str | None) -> str:
    """Дата публикации из выдачи → YYYY-MM-DD (Tavily отдаёт RFC 2822: 'Thu, 08 Oct 2026 12:00:00 GMT')."""
    if not raw:
        return ""
    raw = raw.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return ""


def date_from_url(url: str) -> str:
    """Дата публикации часто зашита в URL новостей: …/2026/10/4/…, …-2026-10-04/, /20261004/."""
    m = re.search(r"(20\d{2})[/-](0?[1-9]|1[0-2])[/-](3[01]|[12]\d|0?[1-9])(?!\d)", url) or re.search(
        r"/(20\d{2})(0[1-9]|1[0-2])(3[01]|[12]\d|0[1-9])/", url)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""


def _ddg_html(query: str, max_results: int, timelimit: str | None) -> list[dict]:
    import html
    from urllib.parse import parse_qs, unquote, urlparse

    import requests

    data = {"q": query}
    if timelimit:
        data["df"] = timelimit
    resp = requests.post("https://html.duckduckgo.com/html/", data=data, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
                                                "(KHTML, like Gecko) Chrome/128 Safari/537.36"})
    resp.raise_for_status()
    page = resp.text
    links = re.findall(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    out = []
    for i, (href, title) in enumerate(links[:max_results]):
        if "duckduckgo.com/l/?" in href:  # редирект-обёртка → реальный URL
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        if "duckduckgo.com/y.js" in href:  # реклама
            continue
        clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()  # noqa: E731
        published = date_from_url(href)
        out.append({"title": clean(title), "url": href, "snippet": clean(snippets[i]) if i < len(snippets) else "",
                    "published": published})
    return out


def web_search(query: str, topic: str = "news", days: int | None = 30, max_results: int | None = None,
               settings: Settings | None = None) -> tuple[list[WebResult], dict]:
    """Возвращает (отфильтрованные результаты, статистика фильтрации)."""
    s = settings or get_settings()
    n = max_results or s.web_max_results
    # Запрашиваем с запасом: часть результатов уйдёт под фильтр
    fetch_n = n * 2
    provider = "tavily" if s.tavily_api_key else "duckduckgo"
    try:
        raw = _tavily(query, fetch_n, topic, days, s.tavily_api_key) if s.tavily_api_key else _ddg(query, fetch_n, topic, days)
    except Exception as e:
        if provider == "tavily":
            log.warning("Tavily недоступен (%s), переключаюсь на DuckDuckGo", e)
            provider = "duckduckgo"
            raw = _ddg(query, fetch_n, topic, days)
        else:
            raise

    kept, dropped, seen = [], [], set()
    for r in raw:
        if not r["url"] or r["url"] in seen:
            continue
        seen.add(r["url"])
        v = assess(r["url"], r["title"])
        if v.blocked:
            dropped.append({"url": r["url"], "reason": v.reason})
            continue
        published = iso_date(r["published"]) or date_from_url(r["url"])
        kept.append(WebResult(r["title"], r["url"], (r["snippet"] or "")[:1200], published,
                              v.name, v.tier, v.tier_label))
    kept.sort(key=lambda w: w.tier)  # стабильная сортировка: внутри уровня сохраняем порядок релевантности
    return kept[:n], {"provider": provider, "dropped": dropped, "raw_count": len(raw)}


def format_results(results: list[WebResult], stats: dict) -> str:
    if not results:
        return f"Веб-поиск ({stats['provider']}) не дал надёжных результатов."
    lines = [f"Провайдер: {stats['provider']}. Отфильтровано как жёлтая пресса/UGC/кликбейт: {len(stats['dropped'])}.", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"--- Результат {i} {r.label()} ({r.tier_label})")
        lines.append(f"Заголовок: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(r.snippet)
        lines.append("")
    return "\n".join(lines)

"""Детерминированная проверка ответа: цитаты и числа сверяются с тем, что агент реально получил.

* Цитаты. Каждая метка в ответе — [Отчёт EIA STEO, октябрь 2026, стр. 7], [Источник: Reuters, web, …],
  [Расчёт: …] — должна соответствовать фрагменту отчёта (отчёт + месяц + страница), который был
  найден в этом ходе, веб-источнику из выдачи поиска или вызванному расчётному инструменту.
* Числа. Числа из текста ответа (кроме годов и мелких порядковых) ищутся в текстах найденных
  фрагментов, сниппетах веб-выдачи и выводе инструментов с точностью до разрядов ответа.
  Ненайденные — не обязательно ошибка (арифметика аналитика, оценка), но их видно пользователю.

Это не замена фактчекингу, а дешёвый страховочный слой от выдуманных страниц, источников и цифр.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CALC_TOOLS = {"forecast_oil_price", "supply_shock_scenario"}

_BRACKET_RE = re.compile(r"\[([^\[\]]{3,500})\](?!\()")
_REPORT_RE = re.compile(
    r"Отч[её]т\s+(?P<report>[^,\]]+?),\s*(?P<date>[а-яё]+\s+\d{4})(?:\s*,\s*стр\.\s*(?P<pages>[\d\s,–\-]+))?",
    re.IGNORECASE,
)
_PAGES_ONLY_RE = re.compile(r"^(?:там же,?\s*)?стр\.\s*(?P<pages>[\d\s,–\-]+)", re.IGNORECASE)
_WEB_RE = re.compile(r"(?:Источник:\s*)?(?P<name>[^,;]+?),\s*web\b", re.IGNORECASE)
_CALC_RE = re.compile(r"Расч[её]т:\s*(?P<what>.+)", re.IGNORECASE)

_NUMBER_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
_NOISE_RES = [
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:/\d{2})?\b"),  # ISO-даты
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),  # 06.10.2026
    re.compile(r"\b\d{1,2}[–-]\d{1,2}\.\d{2}\b"),  # 13–14.10
    re.compile(r"\b[1-4]\s?(?:кв\.|квартал\w*|К|Q)\s?\d{2,4}\b", re.IGNORECASE),  # 2К27, 4 кв. 2026
    re.compile(r"\b(?:Q|К)[1-4]\s?\d{2,4}\b"),
    re.compile(r"\b\d+[- ]?(?:й|го|м|му|я)\b"),  # порядковые: 2-м, 13-го
    # месяц + двузначный год: «дек. 26», «янв 27»
    re.compile(r"\b(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\.?\s*['’]?\d{2}\b", re.IGNORECASE),
]


@dataclass
class Evidence:
    """Что агент получил за ход: фрагменты отчётов, веб-выдача, вызовы и вывод инструментов."""

    report_chunks: list[dict] = field(default_factory=list)  # {report_short, date_human, page, text}
    web_sources: list[dict] = field(default_factory=list)  # {name, url, title, snippet}
    tool_names: set[str] = field(default_factory=set)
    tool_outputs: list[str] = field(default_factory=list)


def _pages(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in re.split(r"[,\s]+(?=\d)", raw.strip()):
        part = part.strip(" ,")
        if m := re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part):
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a < 30:
                out.extend(range(a, b + 1))
                continue
        if part.isdigit():
            out.append(int(part))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().replace("ё", "е"))


def parse_citations(answer: str) -> list[dict]:
    """Разбирает метки источников. Возвращает [{kind, raw, report, date, pages | name | what}]."""
    found: list[dict] = []
    last_report: tuple[str, str] | None = None
    for m in _BRACKET_RE.finditer(answer):
        group = m.group(1).strip()
        if not re.match(r"(Отч[её]т|Источник|Расч[её]т|там же|стр\.)", group, re.IGNORECASE):
            continue
        for part in (p.strip() for p in group.split(";")):
            if not part:
                continue
            if rm := _REPORT_RE.search(part):
                last_report = (rm.group("report").strip(), rm.group("date").strip())
                found.append({"kind": "report", "raw": part, "report": last_report[0], "date": last_report[1],
                              "pages": _pages(rm.group("pages"))})
            elif (pm := _PAGES_ONLY_RE.match(part)) and last_report:
                found.append({"kind": "report", "raw": part, "report": last_report[0], "date": last_report[1],
                              "pages": _pages(pm.group("pages"))})
            elif cm := _CALC_RE.match(part):
                found.append({"kind": "calc", "raw": part, "what": cm.group("what").strip()})
            elif wm := _WEB_RE.search(part):
                found.append({"kind": "web", "raw": part, "name": wm.group("name").strip()})
    return found


def _key(s: str) -> str:
    """Только буквы и цифры: «Arab News» ≡ arabnews.com, «Oil & Gas 360» ≡ oilandgas360 (почти)."""
    return re.sub(r"[^a-zа-я0-9]", "", _norm(s))


def _host(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()


_SOURCE_LINE_RE = re.compile(r"^\s*[-*•]?\s*(?P<name>[^:\n—]+?)\s*[:—–-]\s*<?(?P<url>https?://[^\s>)]+)", re.M)


def source_aliases(answer: str) -> dict[str, str]:
    """Название издания → домен из раздела «Источники» ответа («Quartz, 2026-10-02: https://qz.com/…»).

    Издания часто называются не так, как их домен (Quartz — qz.com), а метка в тексте содержит
    только название; ссылку модель приводит в списке источников — по ней и сопоставляем."""
    aliases = {}
    for m in _SOURCE_LINE_RE.finditer(answer):
        name = re.split(r"[,(]", m.group("name"))[0]
        if len(key := _key(name)) > 1:
            aliases[key] = _host(m.group("url"))
    return aliases


def _web_matches(name: str, sources: list[dict], aliases: dict[str, str] | None = None) -> bool:
    cited = _norm(name)
    parts = [_key(p) for p in re.split(r"\s*/\s*|\s+со ссылкой на\s+|\s*\(|\)", cited)]
    parts = [p for p in parts if len(p) > 1]
    hosts = {aliases[p] for p in parts if aliases and p in aliases}
    for s in sources:
        cands = {_key(s.get("name", ""))}
        if url := s.get("url"):
            host = _host(url)
            if host in hosts:
                return True
            cands |= {_key(host), _key(host.rsplit(".", 1)[0])}
        for c in filter(lambda x: len(x) > 1, cands):
            if any(c in p or p in c for p in parts):
                return True
    return False


def verify_citations(answer: str, ev: Evidence) -> dict:
    retrieved = {(_norm(c["report_short"]), _norm(c["date_human"]), int(c["page"]))
                 for c in ev.report_chunks if c.get("page")}
    reports = {(r, d) for r, d, _ in retrieved}
    aliases = source_aliases(answer)
    items = []
    for c in parse_citations(answer):
        if c["kind"] == "report":
            key = (_norm(c["report"]), _norm(c["date"]))
            if key not in reports:
                status, detail = "not_found", "отчёт с такой датой не встречался в найденных фрагментах"
            elif not c["pages"]:
                status, detail = "ok", "отчёт найден (страница не указана)"
            else:
                missing = [p for p in c["pages"] if (*key, p) not in retrieved]
                status = "ok" if not missing else "page_not_retrieved"
                detail = "страницы найдены" if not missing else f"нет среди найденных фрагментов: стр. {', '.join(map(str, missing))}"
        elif c["kind"] == "web":
            ok = _web_matches(c["name"], ev.web_sources, aliases)
            status, detail = ("ok", "источник есть в веб-выдаче") if ok else ("not_found", "источника нет в веб-выдаче")
        else:
            ok = bool(ev.tool_names & CALC_TOOLS)
            status, detail = ("ok", "расчётный модуль вызывался") if ok else ("not_found", "расчётный модуль не вызывался")
        items.append({**{k: v for k, v in c.items() if k != "pages"}, "pages": c.get("pages", []),
                      "status": status, "detail": detail})
    return {"total": len(items), "verified": sum(i["status"] == "ok" for i in items), "items": items}


def _to_float(token: str) -> tuple[float, int]:
    token = token.replace(",", ".")
    decimals = len(token.split(".")[1]) if "." in token else 0
    return float(token), decimals


def _evidence_numbers(ev: Evidence) -> set[float]:
    texts = [c.get("text", "") for c in ev.report_chunks]
    texts += [f"{s.get('title', '')} {s.get('snippet', '')}" for s in ev.web_sources]
    texts += ev.tool_outputs
    nums: set[float] = set()
    for t in texts:
        for tok in _NUMBER_RE.findall(t or ""):
            try:
                nums.add(_to_float(tok)[0])
            except ValueError:
                continue
    return nums


def _strip_noise(text: str) -> str:
    text = _BRACKET_RE.sub(" ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)  # цели markdown-ссылок
    for rx in _NOISE_RES:
        text = rx.sub(" ", text)
    return text


def verify_numbers(answer: str, ev: Evidence, max_examples: int = 12) -> dict:
    pool = _evidence_numbers(ev)
    total, grounded, ungrounded = 0, 0, []
    # Раздел «Источники» не проверяем: там только перечень документов и страниц
    body = re.split(r"\n#+\s*Источники|\n\*\*Источники\*\*", answer)[0]
    for line in body.splitlines():
        clean = _strip_noise(line)
        for m in _NUMBER_RE.finditer(clean):
            try:
                value, dec = _to_float(m.group())
            except ValueError:
                continue
            if dec == 0 and (value < 10 or 1990 <= value <= 2100):
                continue  # порядковые, кварталы, годы
            total += 1
            # Совпадение с точностью до разряда ответа: 109,6 подтверждается значением 109.55
            tol = 0.5 * 10 ** (-dec) + 1e-9
            if any(abs(v - value) <= tol for v in pool):
                grounded += 1
            elif len(ungrounded) < max_examples:
                ctx = re.sub(r"\s+", " ", _BRACKET_RE.sub("", line)).strip(" -*|#")
                ungrounded.append({"value": m.group(), "context": ctx[:220]})
    return {"total": total, "grounded": grounded, "ungrounded": ungrounded}


def verify_answer(answer: str, ev: Evidence) -> dict:
    return {"citations": verify_citations(answer, ev), "numbers": verify_numbers(answer, ev)}

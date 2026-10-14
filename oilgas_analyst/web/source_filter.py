"""Фильтр источников веб-поиска: отсекаем жёлтую прессу, ранжируем по надёжности.

Три механизма:
1. Чёрный список доменов (таблоиды, конспирология, соцсети/агрегаторы без редакции).
2. Эвристика кликбейта по заголовку (капслок, «шок», «you won't believe» и т.п.).
3. Уровни доверия для оставшихся источников — агент видит уровень в ответе инструмента
   и при конфликте данных отдаёт приоритет более надёжному.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Уровень 1 — первичные/официальные источники (агентства, регуляторы, министерства)
OFFICIAL = {
    "opec.org": "OPEC", "iea.org": "IEA", "eia.gov": "EIA", "energy.gov": "US DoE",
    "worldbank.org": "World Bank", "imf.org": "IMF", "bis.org": "BIS",
    "treasury.gov": "US Treasury", "ec.europa.eu": "European Commission",
    "consilium.europa.eu": "Council of the EU", "minenergo.gov.ru": "Минэнерго РФ",
    "minfin.gov.ru": "Минфин РФ", "cbr.ru": "Банк России", "government.ru": "Правительство РФ",
    "cmegroup.com": "CME Group", "ice.com": "ICE", "jodidata.org": "JODI", "ief.org": "IEF",
}

# Уровень 2 — крупные информагентства и деловые/отраслевые издания с редакционной политикой
REPUTABLE = {
    "reuters.com": "Reuters", "bloomberg.com": "Bloomberg", "ft.com": "Financial Times",
    "wsj.com": "WSJ", "apnews.com": "AP", "cnbc.com": "CNBC", "economist.com": "The Economist",
    "spglobal.com": "S&P Global", "argusmedia.com": "Argus", "woodmac.com": "Wood Mackenzie",
    "rystadenergy.com": "Rystad Energy", "kpler.com": "Kpler", "nytimes.com": "NYT",
    "theguardian.com": "The Guardian", "bbc.com": "BBC", "bbc.co.uk": "BBC",
    "marketwatch.com": "MarketWatch", "investing.com": "Investing.com",
    "oilprice.com": "OilPrice.com", "upstreamonline.com": "Upstream", "offshore-mag.com": "Offshore",
    "hellenicshippingnews.com": "Hellenic Shipping News", "naturalgasintel.com": "NGI",
    "interfax.ru": "Интерфакс", "interfax.com": "Interfax", "tass.ru": "ТАСС", "tass.com": "TASS",
    "ria.ru": "РИА Новости", "kommersant.ru": "Коммерсантъ", "vedomosti.ru": "Ведомости",
    "rbc.ru": "РБК", "forbes.ru": "Forbes Russia", "neftegaz.ru": "Neftegaz.RU",
    "oilcapital.ru": "Нефть и Капитал", "bigpowernews.ru": "BigpowerNews", "frankmedia.ru": "Frank Media",
    "finmarket.ru": "Финмаркет", "prime1.ru": "ПРАЙМ", "1prime.ru": "ПРАЙМ",
}

# Чёрный список: таблоиды, конспирология, UGC-платформы
BLOCKED = {
    # англоязычная жёлтая пресса
    "dailymail.co.uk", "thesun.co.uk", "the-sun.com", "mirror.co.uk", "express.co.uk",
    "dailystar.co.uk", "nypost.com", "tmz.com", "buzzfeed.com", "dailyexpress.co.uk",
    "thesun.ie", "metro.co.uk", "inquisitr.com", "radaronline.com", "ladbible.com",
    "zerohedge.com", "infowars.com", "naturalnews.com", "theonion.com", "babylonbee.com",
    # русскоязычная жёлтая пресса и сомнительные агрегаторы
    "life.ru", "kp.ru", "mk.ru", "tsargrad.tv", "ren.tv", "pravda.ru", "dni.ru",
    "express-novosti.ru", "politros.com", "topcor.ru", "pikabu.ru", "yaplakal.com",
    "dzen.ru", "zen.yandex.ru", "nation-news.ru", "rusvesna.su",
    # соцсети и UGC — без редакционной проверки
    "t.me", "vk.com", "ok.ru", "facebook.com", "x.com", "twitter.com", "tiktok.com",
    "reddit.com", "quora.com", "youtube.com", "instagram.com", "medium.com",
}

CLICKBAIT_PATTERNS = [
    r"\bшок\w*", r"сенсаци\w*", r"вы не поверите", r"срочно!", r"скандал\w*",
    r"you won'?t believe", r"\bshock(ing)?\b", r"jaw[- ]dropping", r"mind[- ]blowing",
    r"\bslams?\b", r"\bdestroys?\b", r"!!+", r"\?\?+",
]
_CLICKBAIT_RE = re.compile("|".join(CLICKBAIT_PATTERNS), re.IGNORECASE)


@dataclass
class SourceVerdict:
    domain: str
    name: str
    tier: int  # 1 — официальный, 2 — надёжный, 3 — непроверенный
    blocked: bool
    reason: str = ""

    @property
    def tier_label(self) -> str:
        return {1: "официальный источник", 2: "надёжное СМИ", 3: "непроверенный источник"}[self.tier]


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _match(domain: str, table) -> str | None:
    """Совпадение домена или любого его родительского домена (news.bbc.co.uk → bbc.co.uk)."""
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in table:
            return candidate
    return None


def is_clickbait(title: str) -> bool:
    if not title:
        return False
    if _CLICKBAIT_RE.search(title):
        return True
    letters = [c for c in title if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    return len(letters) > 20 and caps_ratio > 0.6


def assess(url: str, title: str = "") -> SourceVerdict:
    domain = domain_of(url)
    if (hit := _match(domain, BLOCKED)) is not None:
        return SourceVerdict(domain, domain, 3, True, f"домен {hit} в чёрном списке (жёлтая пресса/UGC)")
    if (hit := _match(domain, OFFICIAL)) is not None:
        return SourceVerdict(domain, OFFICIAL[hit], 1, False)
    if (hit := _match(domain, REPUTABLE)) is not None:
        return SourceVerdict(domain, REPUTABLE[hit], 2, False)
    if is_clickbait(title):
        return SourceVerdict(domain, domain, 3, True, "кликбейтный заголовок")
    return SourceVerdict(domain, domain, 3, False)


def blocked_domains() -> list[str]:
    return sorted(BLOCKED)

import pytest

from oilgas_analyst.web.source_filter import assess, domain_of, is_clickbait


@pytest.mark.parametrize("url", [
    "https://www.dailymail.co.uk/news/oil.html",
    "https://www.kp.ru/daily/123/",
    "https://t.me/some_channel/42",
    "https://life.ru/p/1",
    "https://m.vk.com/wall-1",
])
def test_tabloids_and_ugc_blocked(url):
    assert assess(url, "Oil prices").blocked


@pytest.mark.parametrize("url,tier,name", [
    ("https://www.opec.org/press.html", 1, "OPEC"),
    ("https://www.eia.gov/outlooks/steo/", 1, "EIA"),
    ("https://www.reuters.com/business/energy/x", 2, "Reuters"),
    ("https://www.interfax.ru/business/1", 2, "Интерфакс"),
    ("https://news.bbc.co.uk/x", 2, "BBC"),  # поддомен известного издания
])
def test_tiers(url, tier, name):
    v = assess(url, "OPEC+ agrees to extend cuts")
    assert not v.blocked and v.tier == tier and v.name == name


def test_unknown_domain_is_tier3_unless_clickbait():
    assert assess("https://some-blog.example/post", "Brent rises on supply concerns").tier == 3
    assert assess("https://some-blog.example/post", "ШОК! Нефть рухнет завтра").blocked


def test_clickbait_heuristic():
    assert is_clickbait("You won't believe what OPEC did")
    assert is_clickbait("OIL PRICES WILL EXPLODE TOMORROW SAYS INSIDER")
    assert not is_clickbait("OPEC+ keeps output policy unchanged in November")


def test_reputable_name_is_not_overridden_by_clickbait_rule():
    # Надёжный домен не блокируется эвристикой заголовка
    assert not assess("https://www.reuters.com/x", "Oil SLAMS lower!!").blocked


def test_domain_of():
    assert domain_of("https://www.Reuters.com/a") == "reuters.com"


@pytest.mark.parametrize("url,expected", [
    ("https://www.worldoil.com/news/2026/10/4/opec-holds", "2026-10-04"),
    ("https://www.reuters.com/business/energy/opec-steady-2026-10-04/", "2026-10-04"),
    ("https://english.news.cn/20261004/2f1ee4/c.html", "2026-10-04"),
    ("https://www.kommersant.ru/theme/555", ""),
])
def test_date_from_url(url, expected):
    from oilgas_analyst.web.search import date_from_url

    assert date_from_url(url) == expected

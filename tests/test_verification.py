"""Проверка цитат и чисел на формах меток, которые реально встречаются в ответах (см. demo/transcripts.md)."""

from oilgas_analyst.verification import Evidence, parse_citations, verify_answer


def _chunk(page, text="", report="EIA STEO", date="октябрь 2026"):
    return {"report_short": report, "date_human": date, "page": page, "text": text}


EV = Evidence(
    report_chunks=[
        _chunk(37, "Total world consumption 102.43 104.64 Table 3a"),
        _chunk(17, "World liquid fuels consumption growth"),
        _chunk(6, "Brent will average $105/b in 4Q26"),
        _chunk(7, "fall to $87/b in 2Q27 and $74/b in 4Q27"),
        _chunk(3, "Brent $74/b in 2027", date="сентябрь 2026"),
    ],
    web_sources=[
        {"name": "Reuters", "url": "https://www.reuters.com/business/energy/opec-x-2026-10-04/",
         "title": "OPEC+ keeps output", "snippet": "about 2 million bpd of cuts remain"},
        {"name": "oilandgas360.com", "url": "https://www.oilandgas360.com/opec-holds/", "title": "", "snippet": ""},
    ],
    tool_names={"forecast_oil_price", "web_search"},
    tool_outputs=["окт 2026 | 117.4 | 97.6 – 138.9"],
)


def test_parse_multi_part_labels():
    answer = ("Спрос [Отчёт EIA STEO, октябрь 2026, стр. 37; стр. 17] и прогноз [Отчёт EIA STEO, сентябрь 2026, "
              "стр. 3; там же, стр. 43]. Квоты [Источник: Reuters, web, 2026-10-04; AP, web, 2026-10-04]. "
              "Модель [Расчёт: модуль прогноза, SARIMA+ETS]. Ссылка [Reuters](https://x.y) не метка.")
    cites = parse_citations(answer)
    kinds = [c["kind"] for c in cites]
    assert kinds == ["report", "report", "report", "report", "web", "web", "calc"]
    assert cites[1] == {**cites[1], "report": "EIA STEO", "date": "октябрь 2026", "pages": [17]}
    assert cites[3]["date"] == "сентябрь 2026" and cites[3]["pages"] == [43]
    assert [c["name"] for c in cites if c["kind"] == "web"] == ["Reuters", "AP"]


def test_page_ranges_and_statuses():
    answer = ("Квартальный путь [Отчёт EIA STEO, октябрь 2026, стр. 6–7] и годовые цифры "
              "[Отчёт EIA STEO, октябрь 2026, стр. 58]. Чужой отчёт [Отчёт OPEC MOMR, сентябрь 2026, стр. 12].")
    items = verify_answer(answer, EV)["citations"]["items"]
    assert [(i["pages"], i["status"]) for i in items] == [
        ([6, 7], "ok"), ([58], "page_not_retrieved"), ([12], "not_found")]


def test_web_and_calc_matching():
    answer = ("Решение [Источник: oilandgas360.com / World Oil, web, 2026-10-05]; "
              "неизвестный [Источник: Daily Planet, web]. Расчёт [Расчёт: сценарный модуль].")
    items = verify_answer(answer, EV)["citations"]["items"]
    assert [i["status"] for i in items] == ["ok", "not_found", "ok"]


def test_numbers_grounding_skips_years_and_ordinals():
    answer = ("Спрос в 2026 году — 102,43 млн б/с, в 2027 — 104,64 [Отчёт EIA STEO, октябрь 2026, стр. 37]. "
              "Во 2К27 Brent упадёт до 87 $/барр., модель даёт 117,4 (97,6–138,9). "
              "Профицит около 4,9 млн б/с (моя арифметика), 13–14.10 цена была 106,5.")
    n = verify_answer(answer, EV)["numbers"]
    values = [u["value"] for u in n["ungrounded"]]
    assert values == ["4,9", "106,5"]  # 2026/2027, «2К27», даты и страницы не считаются
    assert (n["grounded"], n["total"]) == (6, 8)


def test_sources_section_is_not_checked_for_numbers():
    answer = "Brent 105 [Отчёт EIA STEO, октябрь 2026, стр. 6].\n\n## Источники\n- EIA STEO, стр. 5, 6, 7, 58, 999"
    assert verify_answer(answer, EV)["numbers"]["total"] == 1


def test_rounded_numbers_are_grounded():
    ev = Evidence(tool_outputs=["production 109.55 mb/d, price 87.4"])
    n = verify_answer("Добыча около 109,6 млн б/с, цена ~87 $/барр., а 110,2 — нет.", ev)["numbers"]
    assert (n["grounded"], n["total"]) == (2, 3) and n["ungrounded"][0]["value"] == "110,2"


def test_web_name_matches_domain_without_spaces_and_month_years_are_ignored():
    ev = Evidence(web_sources=[{"name": "arabnews.com", "url": "https://www.arabnews.com/energy/x"}])
    v = verify_answer("Иран сменил тактику [Источник: Arab News, web, 2026-10-14]. "
                      "Фьючерсы дек. 26 / янв. 27 — 106,35 / 103,71.", ev)
    assert v["citations"]["verified"] == 1
    assert [u["value"] for u in v["numbers"]["ungrounded"]] == ["106,35", "103,71"]


def test_publisher_name_resolved_through_sources_section():
    ev = Evidence(web_sources=[{"name": "qz.com", "url": "https://qz.com/opec-plus-oil-capacity-review"}])
    answer = ("Оценку мощностей отложили [Источник: Quartz, web, 2026-10-02].\n\n## Источники\n"
              "- Quartz, 2026-10-02: https://qz.com/opec-plus-oil-capacity-review\n"
              "- Daily Planet: https://dailyplanet.example/x\n"
              "И ещё [Источник: Daily Planet, web] — ссылки нет в выдаче.")
    items = verify_answer(answer, ev)["citations"]["items"]
    assert [i["status"] for i in items] == ["ok", "not_found"]

#!/usr/bin/env python3
"""Прогон демонстрационных сценариев и сохранение транскриптов в demo/transcripts.md.

    python scripts/run_demo.py            # все сценарии
    python scripts/run_demo.py 1 4        # выборочно
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from oilgas_analyst.agent import OilGasAnalyst
from oilgas_analyst.config import get_settings

SCENARIOS = [
    ("Ответ на основе отчёта",
     ["Какой прогноз мирового спроса на жидкие углеводороды на 2026 и 2027 годы даёт EIA в последнем STEO?"]),
    ("Ответ на основе веб-поиска",
     ["Какие решения по добыче приняла ОПЕК+ на последней встрече? Нужны свежие новости."]),
    ("Комбинированный ответ (отчёты + веб)",
     ["Сравни прогноз EIA по цене Brent с тем, где цена находится сейчас. Рынок сейчас выше или ниже прогноза агентства и почему?"]),
    ("Вызов расчётного модуля: прогноз цены",
     ["Спрогнозируй цену Brent на 3 месяца с доверительным интервалом 80%.",
      "А если ОПЕК+ дополнительно сократит добычу на 1,5 млн б/с — какой диапазон цен через полгода?"]),
    ("Запрос вне компетенции",
     ["Посоветуй, как приготовить борщ и какой фильм посмотреть вечером?"]),
]


def main(selected: list[int]) -> int:
    s = get_settings()
    analyst = OilGasAnalyst(s)
    out = Path(__file__).resolve().parent.parent / "demo" / "transcripts.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# Демонстрационные диалоги",
        "",
        f"Сгенерировано скриптом `scripts/run_demo.py` {datetime.now():%Y-%m-%d %H:%M}. "
        f"Модель: `{s.llm_model}` (effort={s.llm_effort}), веб-поиск: "
        f"{'Tavily' if s.tavily_api_key else 'DuckDuckGo'}, эмбеддинги: `{s.embedding_model}`.",
        "",
        "Под каждым ответом — трасса: что нашёл автопоиск по базе отчётов (шаг 1 правила приоритизации), "
        "какие инструменты агент вызвал сам, результат автоматической проверки цитат и чисел и стоимость.",
        "",
    ]
    for i, (title, questions) in enumerate(SCENARIOS, 1):
        if selected and i not in selected:
            continue
        thread = str(uuid.uuid4())
        lines += [f"## {i}. {title}", ""]
        for q in questions:
            print(f"[{i}] {q}", flush=True)
            t0 = time.time()
            turn = analyst.ask(q, thread)
            dt = time.time() - t0
            relevant = [x for x in turn.report_sources if x["score"] >= s.rag_relevance_threshold]
            lines += [f"**Пользователь:** {q}", "", "**Аналитик:**", "", turn.answer, "",
                      "<details><summary>Трасса</summary>", "",
                      f"- Автопоиск по отчётам: {len(turn.report_sources)} фрагментов, релевантных — {len(relevant)}"
                      + (f" (лучший: {turn.report_sources[0]['label']}, score {turn.report_sources[0]['score']:.3f})"
                         if turn.report_sources else "")]
            for tc in turn.tool_calls:
                lines.append(f"- Инструмент `{tc['name']}`: `{json.dumps(tc['args'], ensure_ascii=False)}`")
            for art in turn.artifacts:
                if art.get("kind") == "web" and art.get("dropped"):
                    lines.append("- Фильтр источников отсеял: " + ", ".join(d["url"] for d in art["dropped"]))
            if turn.verification:
                c, n = turn.verification["citations"], turn.verification["numbers"]
                lines.append(f"- Проверка: цитаты подтверждены {c['verified']}/{c['total']}, "
                             f"числа найдены в источниках {n['grounded']}/{n['total']}")
                for item in c["items"]:
                    if item["status"] != "ok":
                        lines.append(f"  - ⚠️ `{item['raw']}` — {item['detail']}")
                for u in n["ungrounded"][:5]:
                    lines.append(f"  - число {u['value']} не найдено в источниках: «{u['context'][:120]}»")
            u = turn.usage
            if u.get("calls"):
                calls = f"{u['calls']} " + ("вызов" if u["calls"] % 10 == 1 and u["calls"] % 100 != 11 else
                                            "вызова" if u["calls"] % 10 in (2, 3, 4) and u["calls"] % 100 not in (12, 13, 14)
                                            else "вызовов")
                lines.append(f"- Стоимость: ${u['cost_usd']:.3f} ({calls} модели; токены вход "
                             f"{u['input_tokens']}, из кэша {u['cache_read']}, выход {u['output_tokens']})")
            lines += [f"- Время ответа: {dt:.0f} с", "", "</details>", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]]))

---
name: oilgas_analyst
description: Нефтегазовый аналитик — ответы по рынку нефти с RAG по отчётам OPEC/EIA/IEA, веб-поиском без жёлтой прессы и прогнозом цен Brent/WTI (SARIMA+ETS, сценарии ОПЕК+).
version: 0.1.0
type: extension
runtime: python3
entry: plugin.py
plugin_api: "2.0"
permissions: [net, tool, route, widget, read_settings, supervised_task]
env_from_settings: [TAVILY_API_KEY]
dependencies:
  - langgraph
  - langchain-core
  - langchain-anthropic
  - langchain-chroma
  - langchain-text-splitters
  - chromadb
  - sentence-transformers
  - pypdf
  - yfinance
  - statsmodels
  - pandas
  - numpy
  - requests
  - tavily-python
  - ddgs
  - pydantic-settings
  - python-dotenv
when_to_use: Owner asks about the oil and gas market — Upstream/Midstream/Downstream, Brent/WTI/Urals prices, OPEC+ decisions, sanctions, supply/demand balance — or asks to forecast an oil price or estimate the price effect of a production cut.
model_experience:
  what_model_sees: Seven tools (ask, search_reports, web_search, forecast, supply_shock, refresh_reports, add_report). `ask` runs the bundled LangGraph analyst and needs a granted ANTHROPIC_API_KEY; without it (subscription-only install) YOU are the analyst — follow the role and source rules in this skill's body, calling search_reports first (English query), then web_search for fresh facts, forecast/supply_shock for price questions, and copy the source labels the tools return next to every figure.
  token_effect: Fixed schema cost of seven tools per round while enabled; tool results are 1–6K tokens (report fragments or a forecast table).
timeout_sec: 300
scheduled_tasks:
  - name: oilgas-refresh-reports
    cron: "0 9 16 * *"
    timezone: Europe/Moscow
    description: Monthly — call the oilgas_analyst refresh_reports tool to fetch the new EIA STEO edition and re-index; then look for the new OPEC MOMR and add it with add_report.
ui_tab:
  tab_id: analyst
  title: Нефтегазовый аналитик
  icon: "🛢"
  render:
    kind: declarative
    schema_version: 1
    components:
      - type: form
        route: ask
        method: POST
        target: answer
        fields:
          - name: question
            label: Вопрос аналитику
            type: text
            placeholder: "Например: что ОПЕК+ решила по добыче и как это повлияет на Brent?"
            required: true
        submit_label: Спросить
      - type: markdown
        target: answer
        path: answer
      - type: form
        route: forecast
        method: POST
        target: forecast
        fields:
          - name: benchmark
            label: Бенчмарк (brent / wti)
            type: text
            placeholder: brent
          - name: horizon_months
            label: Горизонт, мес.
            type: text
            placeholder: "3"
          - name: supply_change_mbd
            label: Шок предложения, млн б/с (минус — сокращение; пусто — без сценария)
            type: text
            placeholder: "-1.5"
        submit_label: Рассчитать
      - type: markdown
        target: forecast
        path: report
---

# Нефтегазовый аналитик

Навык превращает Ouroboros в **Старшего аналитика нефтегазового рынка**. Ядро — пакет
`oilgas_analyst` (LangGraph + Claude), поставляется внутри payload этого навыка.

## Инструменты

| Инструмент | Назначение |
|---|---|
| `ask` | Полный цикл: автопоиск по базе отчётов → (уточняющий RAG / веб / расчёт по решению модели) → ответ с метками источников. Используй для любых отраслевых вопросов. |
| `search_reports` | Только поиск по базе отчётов (OPEC MOMR, EIA STEO, IEA OMR); запрос лучше на английском. |
| `web_search` | Веб-поиск (Tavily → DuckDuckGo) с отсевом жёлтой прессы и соцсетей. |
| `forecast` | Прогноз среднемесячной цены Brent/WTI с доверительным интервалом (SARIMA + ETS, веса по бэктесту). |
| `supply_shock` | Диапазон цены при изменении мирового предложения на N млн б/с. |
| `refresh_reports` | Скачать свежие выпуски EIA STEO и проиндексировать базу. Первый вызов также скачивает модель эмбеддингов (~470 МБ). |
| `add_report` | Добавить PDF-отчёт по прямой ссылке (например, новый OPEC MOMR) и проиндексировать. |

## Роль (если отвечаешь сам, без `ask`)
Ты — Старший аналитик нефтегазового рынка: Upstream, Midstream, Downstream, ценообразование
Brent/WTI/Urals, ОПЕК+, санкции, баланс спроса и предложения. Ответ: краткий вывод → цифры с
единицами и датами → риски и неопределённость → раздел «Источники» (Отчёты / Веб / Расчёты).
Не выдумывай цифры; факт, расчёт и своё суждение различай явно. Прогноз модуля — экстраполяция:
дополни его фундаментальным контекстом из отчётов и новостей. Urals в модуле нет — объясняй через
Brent минус дисконт. Вопросы вне отрасли — вежливый отказ без вызова инструментов; без персональных
инвестиционных рекомендаций.

## Правило источников
1. Сначала база отчётов агентств; 2. если её достаточно — она основной источник;
3. веб — для актуальности (новости, котировки, решения ОПЕК+, санкции);
4. каждая цифра помечена: `[Отчёт EIA STEO, октябрь 2026, стр. 5]`, `[Источник: Reuters, web, 2026-10-12]`,
`[Расчёт: модуль прогноза]`. Если данных нет — аналитик говорит об этом и не выдумывает цифры.

## Первый запуск
После включения вызови `refresh_reports` один раз (несколько минут: модель эмбеддингов + индексация).
OPEC MOMR и IEA OMR закрыты Cloudflare/подпиской от прямого скачивания: найди PDF браузером и
передай прямую ссылку в `add_report`.

## Данные и состояние
Отчёты, индекс Chroma, снимок цен и кэш модели эмбеддингов лежат в state-директории навыка
(`data/state/skills/oilgas_analyst/`). Навык ничего не пишет за её пределами.

## Ключи
- `ANTHROPIC_API_KEY` — нужен только для `ask` (встроенный LangGraph-аналитик). По умолчанию навык
  собирается для режима подписки и ключ не запрашивает (Ouroboros не включает навык, пока не выданы все
  запрошенные ключи): аналитиком выступает сама модель Ouroboros, остальные 6 инструментов ключей LLM не
  требуют. Сборка с ключом: `build_skill.py --with-anthropic-key`.
- `TAVILY_API_KEY` — необязателен; без него веб-поиск идёт через DuckDuckGo.

## Network policy
Обращается только к: `api.anthropic.com` (LLM), `api.tavily.com` / DuckDuckGo (веб-поиск),
`query1.finance.yahoo.com` / `fred.stlouisfed.org` (история цен), `www.eia.gov` (отчёты STEO),
`huggingface.co` (однократная загрузка модели эмбеддингов), а также к URL, явно переданному в
`add_report`. Ключи не логируются и никуда, кроме своих API, не передаются.

# Заметка: эксперимент с Ouroboros (отложен)

Пробовали сделать аналитика навыком (skill) самомодифицирующегося агента
[Ouroboros](https://github.com/razzant/ouroboros) и запускать его на подписках через
[Claudexor](https://github.com/razzant/claudexor). В основную поставку это не вошло: проект
сделан строго по ТЗ (LangGraph + Claude API). Код интеграции сохранён в ветке
`experiment/ouroboros`.

## Что было сделано (ветка `experiment/ouroboros`)
- **Навык для Ouroboros 7.x** (`integrations/ouroboros/skill/`): `SKILL.md` (PluginAPI 2.0) +
  `plugin.py`. Даёт 7 инструментов: `ask`, `search_reports`, `web_search`, `forecast`,
  `supply_shock`, `refresh_reports`, `add_report`. Также виджет с формами вопроса и прогноза и
  ежемесячное `scheduled_task` на обновление базы отчётов.
- **Сборщик** `build_skill.py --install ~/Ouroboros/data --seed`. Payload самодостаточен
  (внутри копия пакета `oilgas_analyst`). `--seed` кладёт в state навыка готовую базу, индекс
  Chroma и модель эмбеддингов.
- **Плагин для Ouroboros 6.x** (`legacy_v6/`): модуль `ouroboros/tools/oilgas.py` и идемпотентный
  установщик.
- **Проверки:** штатный `skill_preflight` Ouroboros 7.6.0 проходит (`ok: true`). Все 104 смоук-теста
  Ouroboros 6.2.0 проходят с установленным плагином. Контракт плагина покрыт pytest.

## Что выяснили
1. **Подписку Claude нельзя поставить главной моделью (Main).** В Ouroboros «сырые» вызовы модели
   через Claudexor есть только у Codex (`docs/architecture/06-agent-core.md`: «The raw-model adapter
   is Codex»). Claude Code подключается только как *агент* для делегированных задач и ревью. Claude
   в роли Main возможен только через API-ключ.
2. **Автопресет субагентов в 7.6.0 устарел.** В `subscription_install_presets.py` для Codex зашиты
   `gpt-5.6-sol`/`gpt-5.6-terra`, а harness Codex уже отдаёт модели GPT-6. Из-за этого падает весь
   пресет (`model_not_in_discovery`), и агент Claude Code не создаётся. Обход: «Use Main for
   reviewers» и ручное добавление субагента с маршрутом `Agent session` → `claude`.
3. **Бюджет и подписки.** Перед каждым вызовом Ouroboros резервирует его оценочную стоимость по
   API-ценам даже на подписке. При `TOTAL_BUDGET=0.01` задачи паузятся ещё до первого вызова модели
   (`budget_scope_paused`, `physical_calls: 0`). Значение `0` означает «без лимита».
4. **Расход квот.** Режим контекста `Max` (сотни тысяч токенов на вызов) быстро выжигает Codex Plus.
   Делегирование в Claude Code съело лимиты Max, а законченного результата не дало.
5. **Ключи навыка.** Ouroboros не включает навык, пока не выданы *все* ключи из `env_from_settings`.
   Поэтому `ANTHROPIC_API_KEY` в манифесте сделан опциональным (`--with-anthropic-key`).
6. **Архитектурный вывод.** В режиме подписки думает модель Ouroboros, а LangGraph-граф не
   задействован. Правило «сначала отчёты» тогда держится только на промпте. Чтобы оно было
   детерминированным, `ask` без ключа должен возвращать готовый пакет доказательств (автопоиск по
   отчётам и веб по необходимости). Это не реализовано.

## Если возвращаться
- Ставить Main на Claude через API-ключ (например, месячные API-кредиты Max), а не через подписку.
- Реализовать «пакет доказательств» из п. 6.
- Переключить контекст чата на `Low`, поставить разумный `Per-task` бюджет.
- Удалить установленный навык: `~/Ouroboros/data/skills/external/oilgas_analyst` и
  `~/Ouroboros/data/state/skills/oilgas_analyst` (около 0.5 ГБ).

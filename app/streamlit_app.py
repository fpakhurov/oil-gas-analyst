"""Streamlit-интерфейс агента «Нефтегазовый аналитик»."""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from oilgas_analyst.agent import OilGasAnalyst
from oilgas_analyst.config import get_settings
from oilgas_analyst.rag.catalog import load_catalog, meta_for
from oilgas_analyst.rag.store import get_store as get_vector_store
from oilgas_analyst.rag.store import indexed_files, ingest_reports
from oilgas_analyst.refresh import last_refresh, refresh_knowledge_base, start_scheduler
from oilgas_analyst.storage import get_store
from oilgas_analyst.web.search import iso_date

st.set_page_config(page_title="Нефтегазовый аналитик", page_icon="🛢️", layout="wide")

MODELS = {"Claude Opus 5.5": "claude-opus-5-5", "Claude Sonnet 5.5": "claude-sonnet-5-5"}
EXAMPLES = [
    "Какой прогноз мирового спроса на нефть на 2027 год даёт EIA?",
    "Что ОПЕК+ решила по добыче на последней встрече?",
    "Сравни оценки баланса рынка в отчётах с последними новостями: профицит или дефицит?",
    "Спрогнозируй цену Brent на 3 месяца",
    "Оцени диапазон цен Brent, если ОПЕК+ сократит добычу на 1,5 млн б/с",
    "Посоветуй рецепт борща",
]
STATUS_ICON = {"ok": "✅", "page_not_retrieved": "⚠️", "not_found": "❌"}


def md(text: str) -> str:
    """Streamlit рендерит $…$ как LaTeX — в ответах про цены доллары нужно экранировать."""
    return (text or "").replace("$", "\\$")

settings = get_settings()
store = get_store(settings)


@st.cache_resource(show_spinner="Загружаю агента…")
def get_analyst(model: str) -> OilGasAnalyst:
    return OilGasAnalyst(settings.model_copy(update={"llm_model": model}))


@st.cache_resource
def background_refresh():
    return start_scheduler(settings)


@st.cache_data(show_spinner=False, max_entries=64)
def pdf_page_png(file: str, page: int) -> bytes | None:
    import pypdfium2 as pdfium

    path = settings.reports_dir / file
    if not path.exists():
        return None
    pdf = pdfium.PdfDocument(str(path))
    try:
        image = pdf[page - 1].render(scale=1.6).to_pil()
    finally:
        pdf.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- отрисовка хода ----------------------------------------------------------

def render_chart(art: dict) -> None:
    hist, fc = art["history"], art["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["dates"], y=hist["values"], name="История (ср. за месяц)", line=dict(color="#555")))
    fig.add_trace(go.Scatter(x=fc["dates"] + fc["dates"][::-1], y=fc["upper"] + fc["lower"][::-1], fill="toself",
                             fillcolor="rgba(31,119,180,0.18)", line=dict(width=0),
                             name=f"Интервал {int(art['confidence'] * 100)}%"))
    main_name = "Ансамбль" if art["kind"] == "forecast" else "Сценарий: центральный"
    fig.add_trace(go.Scatter(x=fc["dates"], y=fc["mean"], name=main_name, line=dict(color="#1f77b4", width=3)))
    for name, values in fc.get("methods", {}).items():
        fig.add_trace(go.Scatter(x=fc["dates"], y=values, name=name, line=dict(dash="dot")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="$/барр.",
                      title=f"{art['benchmark']}: {'прогноз' if art['kind'] == 'forecast' else 'сценарий шока предложения'}",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    if info := art.get("method_info"):
        st.caption("Модели ансамбля: вес — по ошибке скользящего бэктеста (MAPE)")
        st.dataframe([{"Модель": m["method"], "Спецификация": m["details"], "Вес": m["weight"],
                       "MAPE бэктеста": f"{m['backtest_mape'] * 100:.1f}%" if m["backtest_mape"] is not None else "—"}
                      for m in info], hide_index=True, use_container_width=True)
    if a := art.get("assumptions"):
        st.caption(
            f"Допущения: шок {a['supply_change_mbd']:+} млн б/с, компенсация {a['offset_share']:.0%} → эффективно "
            f"{a['effective_shock_mbd']:+} млн б/с ({a['shock_share_pct']:+}% мирового предложения "
            f"{a['global_supply_mbd']:.0f} млн б/с), выход на полную силу за {a['phase_in_months']} мес. "
            "Ценовой эффект: " + ", ".join(f"{k} {v:+}%" for k, v in a["price_impact_pct"].items()))


def badges(turn: dict) -> str:
    parts = []
    v = turn.get("verification") or {}
    if v:
        c, n = v["citations"], v["numbers"]
        icon = "✅" if c["verified"] == c["total"] else "⚠️"
        parts.append(f"{icon} цитаты {c['verified']}/{c['total']}")
        parts.append(f"🔢 числа в источниках {n['grounded']}/{n['total']}")
    if (u := turn.get("usage")) and u.get("calls"):
        parts.append(f"\\${u['cost_usd']:.3f}" + (" (оценка)" if u.get("estimated") else ""))
        parts.append(f"токены {u['input_tokens'] // 1000}k→{u['output_tokens'] // 1000}k, кэш {u['cache_read'] // 1000}k")
    if turn.get("duration_s"):
        parts.append(f"⏱ {turn['duration_s']:.0f} с")
    if turn.get("model"):
        parts.append(turn["model"])
    if turn.get("trace_url"):
        parts.append(f"[трейс в Langfuse ↗]({turn['trace_url']})")
    return " · ".join(parts)


def render_evidence(turn: dict, key: str) -> None:
    chunks, seen = [], set()
    for src in [*(turn.get("report_sources") or []),
                *[s for a in turn.get("artifacts", []) if a.get("kind") == "rag" for s in a.get("sources", [])]]:
        if src.get("label") in seen:
            continue
        seen.add(src.get("label"))
        chunks.append(src)
    cited = {(i.get("report"), i.get("date"), p)
             for i in (turn.get("verification") or {}).get("citations", {}).get("items", [])
             if i["kind"] == "report" for p in i.get("pages", [])}
    if chunks:
        st.markdown("**Фрагменты отчётов** (по убыванию близости к вопросу; 📌 — страница процитирована в ответе)")
        for i, c in enumerate(sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)):
            used = "📌 " if (c.get("report_short"), c.get("date_human"), c.get("page")) in cited else ""
            toggle_key = f"{key}-pdf-{i}"
            # Переключатель перезапускает скрипт — держим раздел раскрытым, пока страница показана
            with st.expander(f"{used}{c['label']} — score {c.get('score', 0):.3f}",
                             expanded=bool(st.session_state.get(toggle_key))):
                st.text(c.get("text", "")[:1500] or "(текст фрагмента не сохранён)")
                if c.get("file") and c.get("page") and st.toggle("Показать страницу PDF", key=toggle_key):
                    png = pdf_page_png(c["file"], int(c["page"]))
                    if png:
                        st.image(png, caption=f"{c['file']}, стр. {c['page']}")
                    else:
                        st.caption("PDF не найден в data/reports")
    for art in turn.get("artifacts", []):
        if art.get("kind") != "web":
            continue
        st.markdown(f"**Веб-поиск** «{art.get('query', '')}» — {art.get('provider', '')}")
        for s in art.get("sources", []):
            date = f", {iso_date(s['published']) or s['published']}" if s.get("published") else ""
            st.markdown(f"- [{md(s['title'] or s['url'])}]({s['url']}) — {s.get('name', '')}{date} · _{s['tier']}_")
        for d in art.get("dropped", []):
            st.markdown(f"- ~~{d['url']}~~ — отсеяно: {d['reason']}")
    if not chunks and not any(a.get("kind") == "web" for a in turn.get("artifacts", [])):
        st.caption("Источники не использовались.")


def render_path(turn: dict) -> None:
    rows = []
    for s in turn.get("steps", []):
        if s["kind"] == "reports_auto":
            rows.append(f"🔎 **Шаг 1 — база отчётов (автоматически):** {s['found']} фрагментов, "
                        f"выше порога релевантности {s['relevant']}; лучший — {s['best']}")
        elif s["kind"] == "note":
            rows.append(f"💬 _{md(s['text'])}_")
        elif s["kind"] == "tool_call":
            rows.append(f"🛠 {md(s['text'])}")
        elif s["kind"] == "tool_result":
            rows.append(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {md(s['text'])}")
    rows.append("📝 Ответ сформирован" + (" → ✅ проверка цитат и чисел" if turn.get("verification") else ""))
    st.markdown("\n\n".join(rows))


def render_verification(turn: dict) -> None:
    v = turn.get("verification") or {}
    if not v:
        st.caption("Проверка не выполнялась.")
        return
    st.caption("Детерминированная сверка ответа с тем, что агент реально нашёл в этом диалоге. "
               "⚠️ — отчёт найден, но такой страницы среди найденных фрагментов нет; ❌ — источника не было.")
    items = v["citations"]["items"]
    if items:
        st.dataframe([{"": STATUS_ICON.get(i["status"], "?"), "Метка": i["raw"], "Результат": i["detail"]}
                      for i in items], hide_index=True, use_container_width=True)
    n = v["numbers"]
    st.markdown(f"**Числа:** {n['grounded']} из {n['total']} найдены в текстах источников и выводе инструментов.")
    if n["ungrounded"]:
        st.markdown("Не найдены (могут быть арифметикой или оценкой аналитика — проверьте):")
        for u in n["ungrounded"]:
            st.markdown(f"- **{u['value']}** — {md(u['context'])}")


def render_turn(turn: dict, idx: int) -> None:
    st.markdown(md(turn["answer"]))
    if b := badges(turn):
        st.caption(b)
    for art in turn.get("artifacts", []):
        if art.get("kind") in ("forecast", "scenario"):
            render_chart(art)
    if turn.get("budget_stopped") or "steps" not in turn:
        return
    tabs = st.tabs(["📚 Доказательства", "🧭 Путь решения", "✅ Проверка", "💭 Рассуждения"])
    with tabs[0]:
        render_evidence(turn, f"t{idx}")
    with tabs[1]:
        render_path(turn)
    with tabs[2]:
        render_verification(turn)
    with tabs[3]:
        if turn.get("reasoning"):
            for r in turn["reasoning"]:
                st.markdown(f"> {md(r)}")
            st.caption("Краткое изложение рассуждений модели (thinking summary), а не полный ход мыслей.")
        else:
            st.caption("Резюме рассуждений нет.")


# --- состояние сессии --------------------------------------------------------

def open_thread(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id or str(uuid.uuid4())
    st.session_state.history = store.get_turns(thread_id) if thread_id else []
    if thread_id and (t := store.get_thread(thread_id)):
        st.session_state.model = t["model"]


if "thread_id" not in st.session_state:
    open_thread(None)
st.session_state.setdefault("model", settings.llm_model)
background_refresh()

# --- боковая панель ------------------------------------------------------------

with st.sidebar:
    st.header("🛢️ Нефтегазовый аналитик")
    models = dict(MODELS)
    if st.session_state.model not in models.values():
        models[st.session_state.model] = st.session_state.model
    labels = list(models)
    current = next(i for i, m in enumerate(models.values()) if m == st.session_state.model)
    picked = models[st.selectbox("Модель", labels, index=current,
                                 help="Opus — глубже анализ; Sonnet — быстрее и вдвое дешевле. "
                                      "Смена модели начинает новый диалог.")]
    if picked != st.session_state.model:
        st.session_state.model = picked
        open_thread(None)
        st.rerun()

    if st.button("➕ Новый диалог", use_container_width=True):
        open_thread(None)
        st.rerun()

    threads = store.list_threads(20)
    if threads:
        st.subheader("Диалоги")
        for t in threads:
            mark = "▸ " if t["thread_id"] == st.session_state.thread_id else ""
            label = f"{mark}{t['title'][:42]}{'…' if len(t['title']) > 42 else ''}"
            if st.button(label, key=f"th-{t['thread_id']}", use_container_width=True,
                         help=f"{t['updated_at'][:16].replace('T', ' ')} UTC · {t['model']}"):
                open_thread(t["thread_id"])
                st.rerun()

    st.subheader("Расходы")
    spent = store.spent_today()
    if settings.daily_budget_usd > 0:
        st.progress(min(spent / settings.daily_budget_usd, 1.0),
                    text=md(f"Сегодня ${spent:.2f} из ${settings.daily_budget_usd:.2f} (UTC)"))
    else:
        st.caption(md(f"Сегодня ${spent:.2f} (лимит выключен)"))
    if settings.tracing_enabled:
        st.markdown(f"Трейсы: [Langfuse ↗]({settings.langfuse_ui_url})")
    else:
        st.caption("Трейсинг выключен (LANGFUSE_ENABLED)")
    st.caption("Веб-поиск: " + ("Tavily" if settings.tavily_api_key else "DuckDuckGo (нет TAVILY_API_KEY)"))

    st.subheader("База отчётов")
    catalog = load_catalog(settings.reports_dir)
    try:
        files = sorted(indexed_files(get_vector_store(settings)))
    except Exception as e:
        files = []
        st.error(f"Векторная БД недоступна: {e}")
    for f in files:
        m = meta_for(f, catalog)
        st.markdown(f"- {m.short_name}, {m.human_date}")
    if not files:
        st.warning("База пуста. Нажмите «Обновить базу» или загрузите PDF.")
    lr = last_refresh(settings)
    when = datetime.fromisoformat(lr["at"]).strftime("%d.%m.%Y %H:%M") + " UTC" if lr else "ещё не было"
    st.caption(f"Автообновление раз в {settings.refresh_interval_days} дн. · последнее: {when}"
               if settings.refresh_interval_days > 0 else f"Автообновление выключено · последнее: {when}")
    if st.button("🔄 Обновить базу сейчас", use_container_width=True):
        with st.spinner("Скачиваю новые выпуски, индексирую, обновляю цены…"):
            res = refresh_knowledge_base(settings)
        new = res.get("ingest", {}).get("indexed", [])
        st.success(f"Новых отчётов: {len(new)}" + (f" ({', '.join(new)})" if new else ""))
    uploaded = st.file_uploader("Добавить PDF-отчёт", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Проиндексировать загруженные", use_container_width=True):
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        for up in uploaded:
            (settings.reports_dir / up.name).write_bytes(up.getvalue())
        with st.spinner("Индексация…"):
            stats = ingest_reports(settings=settings)
        st.success(f"Новых отчётов: {len(stats['indexed'])}, чанков: {stats['chunks']}")
        st.rerun()

# --- чат -------------------------------------------------------------------------

if not settings.anthropic_api_key:
    st.error("Не задан ANTHROPIC_API_KEY — укажите его в .env")

if not st.session_state.history:
    st.markdown("#### Примеры запросов")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="🛢️"):
        render_turn(turn, i)

question = st.chat_input("Спросите о рынке нефти и газа…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🛢️"):
        status = st.status("Анализирую: отчёты → веб → расчёты…", expanded=True)
        thought = status.empty()
        answer_box = st.empty()
        buffer, thinking, final = "", "", None
        try:
            for event in get_analyst(st.session_state.model).stream(question, st.session_state.thread_id):
                if event["type"] == "token":
                    buffer += event["text"]
                    answer_box.markdown(md(buffer) + "▌")
                elif event["type"] == "thinking":
                    thinking += event["text"]
                    thought.caption("💭 " + md(thinking[-280:].replace("\n", " ")))
                elif event["type"] == "step":
                    if event["node"] == "agent" and buffer.strip():
                        status.markdown(f"💬 _{md(buffer.strip())}_")  # реплика модели перед вызовом инструмента
                        buffer = ""
                        answer_box.empty()
                    thinking = ""
                    thought.empty()
                    status.write(md(event["text"]))
                elif event["type"] == "final":
                    final = event["turn"].to_payload()
            status.update(label="Готово" + (f" за {final['duration_s']:.0f} с" if final else ""),
                          state="complete", expanded=False)
        except Exception as e:
            status.update(label="Ошибка", state="error")
            final = {"answer": f"⚠️ Ошибка: {type(e).__name__}: {e}"}
        thought.empty()
        answer_box.empty()
        turn = {"question": question, **(final or {"answer": "⚠️ Пустой ответ"})}
        render_turn(turn, len(st.session_state.history))
    st.session_state.history.append(turn)

"""Streamlit-интерфейс агента «Нефтегазовый аналитик»."""

from __future__ import annotations

import uuid

import plotly.graph_objects as go
import streamlit as st

from oilgas_analyst.agent import OilGasAnalyst
from oilgas_analyst.config import get_settings
from oilgas_analyst.rag.catalog import load_catalog, meta_for
from oilgas_analyst.rag.store import get_store, indexed_files, ingest_reports

st.set_page_config(page_title="Нефтегазовый аналитик", page_icon="🛢️", layout="wide")

EXAMPLES = [
    "Какой прогноз мирового спроса на нефть на 2027 год даёт EIA?",
    "Что ОПЕК+ решила по добыче на последней встрече?",
    "Сравни оценки баланса рынка в отчётах с последними новостями: профицит или дефицит?",
    "Спрогнозируй цену Brent на 3 месяца",
    "Оцени диапазон цен Brent, если ОПЕК+ сократит добычу на 1,5 млн б/с",
    "Посоветуй рецепт борща",
]


MODELS = {"Claude Opus 5.5": "claude-opus-5-5", "Claude Sonnet 5.5": "claude-sonnet-5-5"}


@st.cache_resource(show_spinner="Загружаю агента…")
def get_analyst(model: str) -> OilGasAnalyst:
    return OilGasAnalyst(get_settings().model_copy(update={"llm_model": model}))


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


def render_details(turn: dict) -> None:
    for art in turn.get("artifacts", []):
        if art.get("kind") in ("forecast", "scenario"):
            render_chart(art)
    with st.expander("Как получен ответ: источники и вызовы инструментов"):
        if turn.get("report_sources"):
            st.markdown("**Автопоиск по базе отчётов (шаг 1):**")
            for s in turn["report_sources"]:
                st.markdown(f"- {s['label']} — score {s['score']:.3f}")
        for tc in turn.get("tool_calls", []):
            st.markdown(f"**Инструмент** `{tc['name']}` — `{tc['args']}`")
        for art in turn.get("artifacts", []):
            if art.get("kind") == "web":
                st.markdown(f"**Веб ({art['provider']})**, отфильтровано источников: {len(art['dropped'])}")
                for s in art["sources"]:
                    st.markdown(f"- {s['label']} ({s['tier']}) [{s['title']}]({s['url']})")
                for d in art["dropped"]:
                    st.markdown(f"- ~~{d['url']}~~ — {d['reason']}")
            elif art.get("kind") == "rag":
                st.markdown("**Уточняющий поиск по отчётам:**")
                for s in art["sources"]:
                    st.markdown(f"- {s['label']} — score {s['score']:.3f}")


# --- Sidebar ---
settings = get_settings()
with st.sidebar:
    st.header("🛢️ Нефтегазовый аналитик")
    models = dict(MODELS)
    if settings.llm_model not in models.values():  # модель из .env, которой нет в списке
        models[settings.llm_model] = settings.llm_model
    labels = list(models)
    default = next(i for i, m in enumerate(models.values()) if m == settings.llm_model)
    model = models[st.selectbox("Модель", labels, index=default,
                                help="Opus — глубже анализ; Sonnet — быстрее и вдвое дешевле")]
    if st.session_state.get("model") not in (None, model):
        # История диалога и thinking-блоки привязаны к модели — при смене начинаем новый диалог
        st.session_state.pop("thread_id", None)
        st.session_state.pop("history", None)
    st.session_state["model"] = model
    st.caption(f"effort: `{settings.llm_effort}` · embeddings: `{settings.embedding_model.split('/')[-1]}`")
    st.caption("Веб-поиск: " + ("Tavily" if settings.tavily_api_key else "DuckDuckGo (нет TAVILY_API_KEY)"))

    st.subheader("База отчётов")
    catalog = load_catalog(settings.reports_dir)
    try:
        files = sorted(indexed_files(get_store()))
    except Exception as e:
        files = []
        st.error(f"Векторная БД недоступна: {e}")
    if files:
        for f in files:
            m = meta_for(f, catalog)
            st.markdown(f"- {m.short_name}, {m.human_date}")
    else:
        st.warning("База пуста. Запустите `oilgas download-reports` и проиндексируйте.")

    uploaded = st.file_uploader("Добавить PDF-отчёт", type=["pdf"], accept_multiple_files=True)
    if st.button("Проиндексировать", use_container_width=True):
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        for up in uploaded or []:
            (settings.reports_dir / up.name).write_bytes(up.getvalue())
        with st.spinner("Индексация…"):
            stats = ingest_reports()
        st.success(f"Новых отчётов: {len(stats['indexed'])}, чанков: {stats['chunks']}")
        st.rerun()

    if st.button("Новый диалог", use_container_width=True):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("history", None)
        st.rerun()

# --- Chat ---
if not settings.anthropic_api_key:
    st.error("Не задан ANTHROPIC_API_KEY — укажите его в .env")

st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("history", [])

if not st.session_state.history:
    st.markdown("#### Примеры запросов")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="🛢️"):
        st.markdown(turn["answer"])
        render_details(turn)

question = st.chat_input("Спросите о рынке нефти и газа…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🛢️"):
        with st.spinner("Анализирую: отчёты → веб → расчёты…"):
            try:
                result = get_analyst(model).ask(question, st.session_state.thread_id)
                turn = {"question": question, "answer": result.answer, "tool_calls": result.tool_calls,
                        "artifacts": result.artifacts, "report_sources": result.report_sources}
            except Exception as e:
                turn = {"question": question, "answer": f"⚠️ Ошибка: {e}"}
        st.markdown(turn["answer"])
        render_details(turn)
    st.session_state.history.append(turn)

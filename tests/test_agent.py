"""Граф агента и формирование запроса к Claude — без реальных вызовов API."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

import oilgas_analyst.agent as agent_mod
import oilgas_analyst.tools as tools_mod
from oilgas_analyst.agent import OilGasAnalyst, build_graph, build_llm
from oilgas_analyst.config import Settings
from oilgas_analyst.rag.store import RetrievedChunk
from oilgas_analyst.storage import get_store
from oilgas_analyst.tools import ALL_TOOLS


@pytest.fixture
def settings(tmp_path):
    return Settings(anthropic_api_key="sk-ant-test", data_dir=tmp_path, langfuse_enabled=False)


def test_graph_structure(settings):
    g = build_graph(settings, InMemorySaver()).get_graph()
    assert {"retrieve_reports", "agent", "tools", "verify"} <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert {("__start__", "retrieve_reports"), ("retrieve_reports", "agent"), ("tools", "agent"),
            ("agent", "verify"), ("verify", "__end__")} <= edges


def test_tools_registered():
    assert {t.name for t in ALL_TOOLS} == {"search_reports", "web_search", "forecast_oil_price",
                                           "supply_shock_scenario"}


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_request_payload(settings, model):
    """RAG-контекст хранится отдельным HumanMessage, но в API уходит одним user-ходом без служебных полей."""
    llm = build_llm(settings.model_copy(update={"llm_model": model}))
    msgs = [
        SystemMessage("system"),
        HumanMessage("Вопрос"),
        HumanMessage("[Автоматический поиск…] фрагменты", additional_kwargs={"rag_context": True, "sources": [1]}),
    ]
    payload = llm._get_request_payload(msgs)
    assert payload["model"] == model
    assert payload["output_config"]["effort"] == "medium"
    assert payload["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in payload["betas"]
    assert payload["cache_control"] == {"type": "ephemeral"}  # автоматический prompt caching
    # Opus/Sonnet 5.5: thinking нельзя отключить — допустим только адаптивный режим (или отсутствие параметра)
    assert payload.get("thinking", {"type": "adaptive"})["type"] == "adaptive"
    assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
    assert "sources" not in str(payload["messages"])


def test_prompt_caching_can_be_disabled(settings):
    payload = build_llm(settings.model_copy(update={"llm_prompt_caching": False}))._get_request_payload(
        [HumanMessage("q")])
    assert "cache_control" not in payload


CHUNK = RetrievedChunk("Brent spot price will average $84/b in 2027", 0.9, "[Отчёт EIA STEO, октябрь 2026, стр. 5]",
                       {"source_file": "eia.pdf", "page": 5, "report_short": "EIA STEO", "report_date": "2026-10"})


def _scripted(monkeypatch, script, seen=None):
    it = iter(script)

    class ScriptedLLM:
        def bind_tools(self, tools):
            return RunnableLambda(lambda msgs: (seen.append(msgs) if seen is not None else None) or next(it))

    monkeypatch.setattr(agent_mod, "build_llm", lambda s: ScriptedLLM())
    monkeypatch.setattr(agent_mod, "search_reports", lambda *a, **k: [CHUNK])


def _ai(text, **kw):
    return AIMessage(text, usage_metadata={"input_tokens": 10_000, "output_tokens": 1_000, "total_tokens": 11_000,
                                           "input_token_details": {"cache_read": 6_000, "cache_creation": 2_000}},
                     response_metadata={"model": "claude-opus-5-5"}, **kw)


def test_graph_end_to_end_with_scripted_llm(monkeypatch, settings):
    """Полный ход: автопоиск → forecast_oil_price → ответ → проверка цитат, учёт стоимости, история."""
    monkeypatch.setattr(tools_mod, "run_forecast", lambda *a, **k: ("ПРОГНОЗ Brent 3 мес.: 104.4", {"kind": "forecast"}, None))
    seen = []
    _scripted(monkeypatch, [
        _ai("Посчитаю прогноз.", tool_calls=[{"name": "forecast_oil_price", "args": {"horizon_months": 3}, "id": "c1"}]),
        _ai("EIA ждёт 84 $/барр. в 2027 [Отчёт EIA STEO, октябрь 2026, стр. 5], модель — 104,4 "
            "[Расчёт: модуль прогноза]. Страница 9 [Отчёт EIA STEO, октябрь 2026, стр. 9]."),
    ], seen)
    analyst = OilGasAnalyst(settings, InMemorySaver())
    events = list(analyst.stream("Спрогнозируй Brent на 3 месяца", "t1"))
    turn = events[-1]["turn"]

    assert turn.answer.startswith("EIA ждёт 84")
    assert turn.tool_calls == [{"name": "forecast_oil_price", "args": {"horizon_months": 3}}]
    assert turn.artifacts == [{"tool": "forecast_oil_price", "kind": "forecast"}]
    assert turn.report_sources[0]["date_human"] == "октябрь 2026" and turn.report_sources[0]["text"]
    # шаг 1 приоритизации — до модели; второй вызов видит результат инструмента
    assert isinstance(seen[0][0], SystemMessage) and "Старший аналитик" in seen[0][0].content
    assert "[Автоматический поиск по базе отчётов" in seen[0][-1].content
    assert "ПРОГНОЗ Brent" in seen[1][-1].content
    # путь решения и события стрима
    assert [s["kind"] for s in turn.steps] == ["reports_auto", "note", "tool_call", "tool_result"]
    assert any(e["type"] == "step" and e.get("tool") == "forecast_oil_price" for e in events)
    assert any(e["type"] == "step" and e["node"] == "verify" for e in events)
    # проверка: стр. 5 найдена, стр. 9 — нет; расчёт вызывался; 84 и 104,4 есть в источниках
    c = turn.verification["citations"]
    assert (c["total"], c["verified"]) == (3, 2)
    assert next(i for i in c["items"] if i["pages"] == [9])["status"] == "page_not_retrieved"
    n = turn.verification["numbers"]
    assert n["grounded"] == n["total"] == 2
    # стоимость: 2 вызова Opus 5.5; (2000 некэш × 4 + 2000 × 5 + 6000 × 0.2 + 1000 × 20) / 1e6 = 0.0392
    assert turn.usage["calls"] == 2 and turn.usage["cost_usd"] == pytest.approx(2 * 0.0392)
    store = get_store(settings)
    assert store.spent_today() == pytest.approx(2 * 0.0392)
    assert store.get_turns("t1")[0]["answer"] == turn.answer
    assert store.list_threads()[0]["title"] == "Спрогнозируй Brent на 3 месяца"


def test_daily_budget_stops_before_llm_call(monkeypatch, settings):
    settings = settings.model_copy(update={"daily_budget_usd": 0.01})
    get_store(settings).record_usage("old", "claude-opus-5-5", 1, 1, 0, 0, 0.02)
    called = []
    _scripted(monkeypatch, [], called)
    turn = OilGasAnalyst(settings, InMemorySaver()).ask("Что с Brent?", "t2")
    assert turn.budget_stopped and "дневной лимит" in turn.answer
    assert called == [] and turn.verification == {}


def test_collect_reasoning_and_usage(settings):
    analyst = OilGasAnalyst(settings, InMemorySaver())
    msgs = [
        HumanMessage("ctx", additional_kwargs={"rag_context": True, "relevant": 1,
                                               "sources": [{"label": "[Отчёт X]"}]}),
        AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "t1"}]),
        ToolMessage("res", tool_call_id="t1", name="web_search", artifact={"kind": "web", "sources": []}),
        AIMessage([{"type": "thinking", "thinking": "Сначала отчёты, потом веб"}, {"type": "text", "text": "Ответ"}]),
    ]
    turn = analyst._collect(msgs, {})
    assert turn.answer == "Ответ" and turn.reasoning == ["Сначала отчёты, потом веб"]
    assert turn.tool_calls == [{"name": "web_search", "args": {"query": "q"}}]
    assert turn.steps[0] == {"kind": "reports_auto", "found": 1, "relevant": 1, "best": "[Отчёт X]"}

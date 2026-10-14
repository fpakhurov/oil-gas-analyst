"""Проверки графа и формирования запроса к Claude без реальных вызовов API."""

from langchain_core.messages import HumanMessage, SystemMessage

from oilgas_analyst.agent import OilGasAnalyst, build_graph, build_llm
from oilgas_analyst.config import Settings
from oilgas_analyst.tools import ALL_TOOLS


def _settings(**kw):
    return Settings(anthropic_api_key="sk-ant-test", **kw)


def test_graph_structure():
    g = build_graph(_settings()).get_graph()
    nodes = set(g.nodes)
    assert {"retrieve_reports", "agent", "tools"} <= nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "retrieve_reports") in edges
    assert ("retrieve_reports", "agent") in edges
    assert ("tools", "agent") in edges


def test_tools_registered():
    names = {t.name for t in ALL_TOOLS}
    assert names == {"search_reports", "web_search", "forecast_oil_price", "supply_shock_scenario"}


def test_rag_context_merged_into_user_turn_and_payload():
    """RAG-контекст хранится отдельным HumanMessage, но в API уходит одним user-ходом без служебных полей."""
    llm = build_llm(_settings())
    msgs = [
        SystemMessage("system"),
        HumanMessage("Вопрос"),
        HumanMessage("[Автоматический поиск…] фрагменты", additional_kwargs={"rag_context": True, "sources": [1]}),
    ]
    payload = llm._get_request_payload(msgs)
    assert payload["model"] == "claude-opus-5-5"
    assert payload["output_config"]["effort"] == "medium"
    assert payload["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in payload["betas"]
    # Opus 5.5: thinking нельзя отключить — допустим только адаптивный режим (или отсутствие параметра)
    assert payload.get("thinking", {"type": "adaptive"})["type"] == "adaptive"
    assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
    assert "sources" not in str(payload["messages"])


def test_collect_turn():
    from langchain_core.messages import AIMessage, ToolMessage

    msgs = [
        HumanMessage("ctx", additional_kwargs={"rag_context": True, "sources": [{"label": "[Отчёт X]"}]}),
        AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "t1"}]),
        ToolMessage("res", tool_call_id="t1", name="web_search", artifact={"kind": "web", "sources": []}),
        AIMessage([{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Ответ"}]),
    ]
    turn = OilGasAnalyst._collect(msgs)
    assert turn.answer == "Ответ"
    assert turn.tool_calls == [{"name": "web_search", "args": {"query": "q"}}]
    assert turn.artifacts[0]["kind"] == "web" and turn.report_sources[0]["label"] == "[Отчёт X]"


def test_graph_end_to_end_with_scripted_llm(monkeypatch):
    """Полный проход графа: автопоиск по отчётам → вызов forecast_oil_price → финальный ответ."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    import oilgas_analyst.agent as agent_mod
    import oilgas_analyst.tools as tools_mod
    from oilgas_analyst.rag.store import RetrievedChunk

    chunk = RetrievedChunk("Brent forecast $85/b", 0.9, "[Отчёт EIA STEO, октябрь 2026, стр. 5]",
                           {"source_file": "eia.pdf", "page": 5})
    monkeypatch.setattr(agent_mod, "search_reports", lambda *a, **k: [chunk])
    monkeypatch.setattr(tools_mod, "run_forecast", lambda *a, **k: ("ПРОГНОЗ Brent 3 мес.", {"kind": "forecast"}, None))

    seen_inputs = []
    script = iter([
        AIMessage("", tool_calls=[{"name": "forecast_oil_price", "args": {"horizon_months": 3}, "id": "c1"}]),
        AIMessage("Brent ≈ 104 $/барр. [Расчёт: модуль прогноза]"),
    ])

    class ScriptedLLM:
        def bind_tools(self, tools):
            return RunnableLambda(lambda msgs: seen_inputs.append(msgs) or next(script))

    monkeypatch.setattr(agent_mod, "build_llm", lambda s: ScriptedLLM())
    turn = OilGasAnalyst(_settings()).ask("Спрогнозируй Brent на 3 месяца")

    assert turn.answer.startswith("Brent ≈ 104")
    assert turn.tool_calls == [{"name": "forecast_oil_price", "args": {"horizon_months": 3}}]
    assert turn.artifacts == [{"tool": "forecast_oil_price", "kind": "forecast"}]
    assert turn.report_sources[0]["label"] == "[Отчёт EIA STEO, октябрь 2026, стр. 5]"
    # Первый вызов LLM: system + вопрос + RAG-контекст (шаг 1 приоритизации идёт до модели)
    first = seen_inputs[0]
    assert isinstance(first[0], SystemMessage) and "Старший аналитик" in first[0].content
    assert "[Автоматический поиск по базе отчётов" in first[-1].content and "стр. 5" in first[-1].content
    # Второй вызов видит результат инструмента
    assert "ПРОГНОЗ Brent" in seen_inputs[1][-1].content


def test_sonnet_model_payload():
    """Sonnet 5.5 — альтернативная модель из UI: тот же контракт запроса (effort, fallback, без prefill)."""
    llm = build_llm(_settings(llm_model="claude-sonnet-5-5"))
    payload = llm._get_request_payload([HumanMessage("q")])
    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["output_config"]["effort"] == "medium" and payload["fallbacks"] == "default"

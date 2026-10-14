"""LangGraph-агент «Нефтегазовый аналитик».

Граф:

    START → retrieve_reports → agent ⇄ tools → END

* retrieve_reports — детерминированный шаг 1 правила приоритизации: перед каждым ответом
  ищем по базе отчётов и кладём найденное в контекст (как часть пользовательского хода).
  Так модель всегда видит данные отчётов первыми и сама решает, достаточно ли их.
* agent — Claude с инструментами: search_reports (уточняющий RAG), web_search,
  forecast_oil_price, supply_shock_scenario. Модель сама решает, когда их вызывать.
* tools — исполнение инструментов (ToolNode), артефакты (источники, данные графиков)
  сохраняются в ToolMessage.artifact и используются UI.

История диалога — append-only (InMemorySaver), это важно для кэширования и
сохранения thinking-блоков между ходами.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.prompts import RAG_CONTEXT_HEADER, SYSTEM_PROMPT
from oilgas_analyst.rag.store import format_chunks, search_reports
from oilgas_analyst.tools import ALL_TOOLS

log = logging.getLogger(__name__)


def build_llm(s: Settings) -> ChatAnthropic:
    kwargs: dict = dict(
        model=s.llm_model,
        max_tokens=s.llm_max_tokens,
        reasoning_effort=s.llm_effort,
        max_retries=3,
        timeout=300,
    )
    if s.anthropic_api_key:
        kwargs["api_key"] = s.anthropic_api_key
    if s.anthropic_base_url:
        kwargs["base_url"] = s.anthropic_base_url
    if s.llm_refusal_fallback:
        # Если классификатор безопасности отклонит запрос (например, ложное срабатывание на
        # словах «санкции»/«эмбарго»), сервер сам повторит его на подходящей модели.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    return ChatAnthropic(**kwargs)


def _is_rag_context(msg: BaseMessage) -> bool:
    return isinstance(msg, HumanMessage) and msg.additional_kwargs.get("rag_context", False)


def build_graph(settings: Settings | None = None, checkpointer=None):
    s = settings or get_settings()
    llm = build_llm(s).bind_tools(ALL_TOOLS)

    def retrieve_reports(state: MessagesState):
        question = state["messages"][-1]
        query = question.text if hasattr(question, "text") else str(question.content)
        try:
            chunks = search_reports(query, k=s.rag_top_k, settings=s)
            body = format_chunks(chunks, s.rag_relevance_threshold)
        except Exception as e:  # пустая/недоступная база не должна ломать диалог
            log.warning("Автопоиск по базе отчётов не удался: %s", e)
            chunks, body = [], f"База отчётов недоступна: {e}"
        relevant = sum(c.score >= s.rag_relevance_threshold for c in chunks)
        hint = (
            "Оцени, достаточно ли этих фрагментов для ответа. Если нет — уточни поиск search_reports "
            "и/или используй web_search. Если вопрос вне компетенции — не используй эти данные."
            if relevant else
            "Релевантных фрагментов не найдено: попробуй уточнить запрос search_reports (по-английски) "
            "или используй web_search, если нужна актуальная информация."
        )
        content = f"{RAG_CONTEXT_HEADER}\n{body}\n{hint}"
        sources = [{"label": c.label, "score": c.score, "file": c.metadata.get("source_file"),
                    "page": c.metadata.get("page")} for c in chunks]
        return {"messages": [HumanMessage(content=content, additional_kwargs={"rag_context": True, "sources": sources})]}

    def agent(state: MessagesState):
        system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().strftime("%d.%m.%Y")))
        response = llm.invoke([system, *state["messages"]])
        return {"messages": [response]}

    g = StateGraph(MessagesState)
    g.add_node("retrieve_reports", retrieve_reports)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_edge(START, "retrieve_reports")
    g.add_edge("retrieve_reports", "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


@dataclass
class AgentTurn:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "args"}]
    artifacts: list[dict] = field(default_factory=list)  # артефакты инструментов (источники, графики)
    report_sources: list[dict] = field(default_factory=list)  # результаты автопоиска по базе


class OilGasAnalyst:
    """Удобная обёртка: один экземпляр — много диалогов (thread_id)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_graph(self.settings)

    def ask(self, question: str, thread_id: str | None = None) -> AgentTurn:
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        before = len(self.graph.get_state(config).values.get("messages", []))
        state = self.graph.invoke({"messages": [HumanMessage(question)]}, config)
        new = state["messages"][before:]
        return self._collect(new)

    @staticmethod
    def _collect(messages: list[BaseMessage]) -> AgentTurn:
        turn = AgentTurn(answer="")
        for m in messages:
            if _is_rag_context(m):
                turn.report_sources = m.additional_kwargs.get("sources", [])
            elif isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    turn.tool_calls.append({"name": tc["name"], "args": tc["args"]})
            elif isinstance(m, ToolMessage) and m.artifact:
                turn.artifacts.append({"tool": m.name, **m.artifact})
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if final is not None:
            turn.answer = final.text.strip()
            if final.response_metadata.get("stop_reason") == "refusal" and not turn.answer:
                turn.answer = "Модель отклонила запрос. Попробуйте переформулировать вопрос."
        return turn

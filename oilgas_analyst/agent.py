"""LangGraph-агент «Нефтегазовый аналитик».

Граф:

    START → retrieve_reports → agent ⇄ tools
                                 └→ verify → END

* retrieve_reports — детерминированный шаг 1 правила приоритизации: перед каждым ответом
  ищем по базе отчётов и кладём найденное в контекст (как часть пользовательского хода).
  Так модель всегда видит данные отчётов первыми и сама решает, достаточно ли их.
* agent — Claude с инструментами: search_reports (уточняющий RAG), web_search,
  forecast_oil_price, supply_shock_scenario. Модель сама решает, когда их вызывать.
  Перед каждым вызовом — проверка дневного бюджета, после — учёт токенов и стоимости.
* tools — исполнение инструментов (ToolNode); артефакты (источники, данные графиков)
  сохраняются в ToolMessage.artifact и в LLM не отправляются.
* verify — детерминированная проверка ответа: цитаты и числа сверяются с тем, что агент
  реально получил в этом ходе (verification.py).

История диалога append-only и хранится в SQLite (LangGraph checkpointer): так работают
prompt caching и thinking-блоки между ходами, а диалоги переживают перезапуск.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from oilgas_analyst import observability as obs
from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.prompts import RAG_CONTEXT_HEADER, SYSTEM_PROMPT
from oilgas_analyst.rag.store import chunk_sources, format_chunks, search_reports
from oilgas_analyst.storage import get_store
from oilgas_analyst.tools import ALL_TOOLS
from oilgas_analyst.translate import translate_reasoning
from oilgas_analyst.verification import Evidence, verify_answer

log = logging.getLogger(__name__)

BUDGET_STOP_TEXT = (
    "⛔ Достигнут дневной лимит расходов на LLM (${spent:.2f} из ${limit:.2f}). Ответ не сформирован — "
    "лимит обнулится в полночь UTC, его можно изменить переменной DAILY_BUDGET_USD."
)


def build_llm(s: Settings) -> ChatAnthropic:
    kwargs: dict = dict(
        model=s.llm_model,
        max_tokens=s.llm_max_tokens,
        reasoning_effort=s.llm_effort,
        max_retries=3,
        timeout=300,
    )
    model_kwargs: dict = {}
    if s.anthropic_api_key:
        kwargs["api_key"] = s.anthropic_api_key
    if s.anthropic_base_url:
        kwargs["base_url"] = s.anthropic_base_url
    if s.llm_refusal_fallback:
        # Если классификатор безопасности отклонит запрос (например, ложное срабатывание на
        # словах «санкции»/«эмбарго»), сервер сам повторит его на подходящей модели.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        model_kwargs["fallbacks"] = "default"
    if s.llm_prompt_caching:
        # Автоматическая точка кэша на конце запроса: инструменты, системный промпт и история
        # переиспользуются между шагами агента (tool loop) и ходами диалога
        model_kwargs["cache_control"] = {"type": "ephemeral"}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatAnthropic(**kwargs)


@lru_cache
def sqlite_checkpointer(path: str):
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class AnalystState(MessagesState):
    verification: dict


def _is_rag_context(msg: BaseMessage) -> bool:
    return isinstance(msg, HumanMessage) and msg.additional_kwargs.get("rag_context", False)


INTERRUPTED_TOOL_TEXT = "Вызов инструмента не выполнен: предыдущий ход был прерван до получения результата."


def heal_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Подставляет результат для вызовов инструментов, оставшихся без ответа.

    Если ход оборвался между решением модели вызвать инструменты и их выполнением (перезапуск
    сервера, перезапуск скрипта Streamlit по клику во время генерации), в истории остаётся
    AIMessage с tool_calls без ToolMessage. API требует tool_result сразу за tool_use, и каждый
    следующий вопрос в таком диалоге падал бы с 400. Чиним только запрос (история не трогается),
    детерминированно — префикс запроса стабилен, prompt caching не ломается.
    """
    healed: list[BaseMessage] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        healed.append(m)
        i += 1
        if not (isinstance(m, AIMessage) and m.tool_calls):
            continue
        answered = set()
        while i < len(messages) and isinstance(messages[i], ToolMessage):
            answered.add(messages[i].tool_call_id)
            healed.append(messages[i])
            i += 1
        for tc in m.tool_calls:
            if tc["id"] not in answered:
                healed.append(ToolMessage(INTERRUPTED_TOOL_TEXT, tool_call_id=tc["id"], name=tc["name"],
                                          status="error"))
    return healed


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Сообщения текущего хода: от последнего вопроса пользователя до конца."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not _is_rag_context(messages[i]):
            return messages[i:]
    return messages


def collect_evidence(messages: list[BaseMessage]) -> Evidence:
    ev = Evidence()
    for m in messages:
        if _is_rag_context(m):
            ev.report_chunks.extend(m.additional_kwargs.get("sources", []))
        elif isinstance(m, AIMessage):
            ev.tool_names |= {tc["name"] for tc in m.tool_calls}
        elif isinstance(m, ToolMessage):
            ev.tool_outputs.append(m.text if hasattr(m, "text") else str(m.content))
            art = m.artifact or {}
            if art.get("kind") == "rag":
                ev.report_chunks.extend(art.get("sources", []))
            elif art.get("kind") == "web":
                ev.web_sources.extend(art.get("sources", []))
    return ev


def build_graph(settings: Settings | None = None, checkpointer=None):
    s = settings or get_settings()
    llm = build_llm(s).bind_tools(ALL_TOOLS)
    store = get_store(s)

    def retrieve_reports(state: AnalystState):
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
        return {"messages": [HumanMessage(content=content, additional_kwargs={
            "rag_context": True, "sources": chunk_sources(chunks), "relevant": relevant})]}

    def agent(state: AnalystState, config):
        if s.daily_budget_usd > 0 and (spent := store.spent_today()) >= s.daily_budget_usd:
            text = BUDGET_STOP_TEXT.format(spent=spent, limit=s.daily_budget_usd)
            return {"messages": [AIMessage(content=text, additional_kwargs={"budget_stop": True})]}
        system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().strftime("%d.%m.%Y")))
        response = llm.invoke([system, *heal_tool_calls(state["messages"])], config)
        model, usage = obs.usage_from_message(response, s.llm_model)
        thread_id = (config.get("configurable") or {}).get("thread_id")
        store.record_usage(thread_id, model, usage.input_tokens, usage.output_tokens,
                           usage.cache_read, usage.cache_write, usage.cost_usd)
        return {"messages": [response]}

    def verify(state: AnalystState):
        turn = _current_turn(state["messages"])
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
        if final is None or final.additional_kwargs.get("budget_stop"):
            return {"verification": {}}
        # Доказательства — весь диалог: источник, найденный на прошлом ходе, модель видит в контексте
        return {"verification": verify_answer(final.text, collect_evidence(state["messages"]))}

    g = StateGraph(AnalystState)
    g.add_node("retrieve_reports", retrieve_reports)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("verify", verify)
    g.add_edge(START, "retrieve_reports")
    g.add_edge("retrieve_reports", "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verify"})
    g.add_edge("tools", "agent")
    g.add_edge("verify", END)
    if checkpointer is None:
        checkpointer = sqlite_checkpointer(str(s.checkpoints_db))
    return g.compile(checkpointer=checkpointer)


@dataclass
class AgentTurn:
    answer: str
    model: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "args"}]
    artifacts: list[dict] = field(default_factory=list)  # артефакты инструментов (источники, графики)
    report_sources: list[dict] = field(default_factory=list)  # результаты автопоиска по базе
    steps: list[dict] = field(default_factory=list)  # путь решения: шаги графа по порядку
    reasoning: list[str] = field(default_factory=list)  # резюме рассуждений Claude на языке вопроса
    reasoning_original: list[str] = field(default_factory=list)  # как вернул API (обычно по-английски)
    verification: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    trace_url: str | None = None
    duration_s: float = 0.0
    budget_stopped: bool = False

    def to_payload(self) -> dict:
        return asdict(self)


def _block_texts(msg: AIMessage, kind: str) -> list[str]:
    if isinstance(msg.content, str):
        return [msg.content] if kind == "text" and msg.content.strip() else []
    key = "thinking" if kind == "thinking" else "text"
    return [b.get(key, "") for b in msg.content
            if isinstance(b, dict) and b.get("type") == kind and (b.get(key) or "").strip()]


def _short(text: str, limit: int = 160) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first[:limit] + ("…" if len(first) > limit else "")


class OilGasAnalyst:
    """Удобная обёртка: один экземпляр — много диалогов (thread_id)."""

    def __init__(self, settings: Settings | None = None, checkpointer=None):
        self.settings = settings or get_settings()
        self.store = get_store(self.settings)
        self.graph = build_graph(self.settings, checkpointer)
        # Перевод резюме рассуждений идёт параллельно ходу и не тормозит граф
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reasoning-translate")

    def ask(self, question: str, thread_id: str | None = None) -> AgentTurn:
        turn = None
        for event in self.stream(question, thread_id):
            if event["type"] == "final":
                turn = event["turn"]
        return turn

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict]:
        """События хода для UI: step | token | thinking | reasoning (перевод резюме) | final(turn)."""
        s = self.settings
        thread_id = thread_id or str(uuid.uuid4())
        callbacks = obs.tracing_callbacks(s)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30,
            "callbacks": callbacks,
            "metadata": obs.trace_metadata(thread_id, s.llm_model),
            "run_name": "oilgas-analyst",
        }
        before = len(self.graph.get_state(config).values.get("messages", []))
        self.store.upsert_thread(thread_id, question, s.llm_model)
        started = time.monotonic()
        segments: list[str] = []
        futures: dict[int, Future] = {}
        emitted: set[int] = set()

        def ready_translations(wait: bool = False) -> Iterator[dict]:
            for i, fut in sorted(futures.items()):
                if i in emitted or not (wait or fut.done()):
                    continue
                try:
                    text, _ = fut.result(timeout=60)
                except Exception:
                    text = segments[i]
                emitted.add(i)
                yield {"type": "reasoning", "index": i, "text": text}

        for mode, chunk in self.graph.stream({"messages": [HumanMessage(question)]}, config,
                                             stream_mode=["updates", "messages"]):
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessageChunk):
                    for kind in ("text", "thinking"):
                        for piece in _stream_pieces(msg, kind):
                            yield {"type": "token" if kind == "text" else "thinking", "text": piece}
            else:
                for node, update in (chunk or {}).items():
                    if node == "agent":
                        for m in (update or {}).get("messages", []):
                            for seg in _block_texts(m, "thinking") if isinstance(m, AIMessage) else []:
                                futures[len(segments)] = self._pool.submit(
                                    translate_reasoning, question, seg, s, self.store, thread_id)
                                segments.append(seg)
                    yield from self._step_events(node, update or {})
            yield from ready_translations()
        yield from ready_translations(wait=True)

        state = self.graph.get_state(config).values
        turn = self._collect(state["messages"][before:], state.get("verification") or {})
        if len(turn.reasoning) == len(segments):
            translated, extra = [], obs.Usage()
            for i, seg in enumerate(segments):
                try:
                    text, u = futures[i].result(timeout=60)
                except Exception:
                    text, u = seg, None
                translated.append(text)
                if u is not None:
                    extra.add(u)
            turn.reasoning_original, turn.reasoning = turn.reasoning, translated
            if extra.calls:
                usage = obs.Usage(**{k: v for k, v in turn.usage.items()})
                usage.add(extra)
                turn.usage = usage.to_dict()
        turn.trace_url = obs.trace_url(s, callbacks)
        turn.duration_s = round(time.monotonic() - started, 1)
        obs.flush(s)
        self.store.add_turn(thread_id, question, turn.to_payload())
        yield {"type": "final", "turn": turn, "thread_id": thread_id}

    def _step_events(self, node: str, update: dict) -> Iterator[dict]:
        if node == "retrieve_reports":
            for m in update.get("messages", []):
                src = m.additional_kwargs.get("sources", [])
                yield {"type": "step", "node": node,
                       "text": f"Поиск по базе отчётов: {len(src)} фрагментов, релевантных — "
                               f"{m.additional_kwargs.get('relevant', 0)}"}
        elif node == "agent":
            for m in update.get("messages", []):
                for tc in getattr(m, "tool_calls", []) or []:
                    yield {"type": "step", "node": node, "tool": tc["name"], "args": tc["args"],
                           "text": _describe_call(tc["name"], tc["args"])}
        elif node == "tools":
            for m in update.get("messages", []):
                yield {"type": "step", "node": node, "tool": m.name, "text": f"↳ {_short(m.text)}"}
        elif node == "verify":
            v = update.get("verification") or {}
            if v:
                c, n = v["citations"], v["numbers"]
                yield {"type": "step", "node": node,
                       "text": f"Проверка: цитаты {c['verified']}/{c['total']}, числа {n['grounded']}/{n['total']}"}

    def _collect(self, messages: list[BaseMessage], verification: dict) -> AgentTurn:
        s = self.settings
        turn = AgentTurn(answer="", model=s.llm_model, verification=verification)
        usage = obs.Usage()
        for m in messages:
            if _is_rag_context(m):
                turn.report_sources = m.additional_kwargs.get("sources", [])
                best = turn.report_sources[0]["label"] if turn.report_sources else "—"
                turn.steps.append({"kind": "reports_auto", "found": len(turn.report_sources),
                                   "relevant": m.additional_kwargs.get("relevant", 0), "best": best})
            elif isinstance(m, AIMessage):
                if m.additional_kwargs.get("budget_stop"):
                    turn.budget_stopped = True
                    continue
                turn.model, u = obs.usage_from_message(m, s.llm_model)
                usage.add(u)
                turn.reasoning.extend(_block_texts(m, "thinking"))
                if m.tool_calls:
                    note = " ".join(_block_texts(m, "text")).strip()
                    if note:
                        turn.steps.append({"kind": "note", "text": note})
                for tc in m.tool_calls:
                    turn.tool_calls.append({"name": tc["name"], "args": tc["args"]})
                    turn.steps.append({"kind": "tool_call", "name": tc["name"], "args": tc["args"],
                                       "text": _describe_call(tc["name"], tc["args"])})
            elif isinstance(m, ToolMessage):
                if m.artifact:
                    turn.artifacts.append({"tool": m.name, **m.artifact})
                turn.steps.append({"kind": "tool_result", "name": m.name, "text": _short(m.text, 240)})
        turn.usage = usage.to_dict()
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if final is not None:
            turn.answer = final.text.strip()
            if final.response_metadata.get("stop_reason") == "refusal" and not turn.answer:
                turn.answer = "Модель отклонила запрос. Попробуйте переформулировать вопрос."
        return turn


def _stream_pieces(chunk: AIMessageChunk, kind: str) -> list[str]:
    if isinstance(chunk.content, str):
        return [chunk.content] if kind == "text" and chunk.content else []
    key = "thinking" if kind == "thinking" else "text"
    return [b.get(key, "") for b in chunk.content
            if isinstance(b, dict) and b.get("type") == kind and b.get(key)]


def _describe_call(name: str, args: dict) -> str:
    if name == "search_reports":
        pub = f" ({args['publisher']})" if args.get("publisher") else ""
        return f"Уточняющий поиск по отчётам{pub}: «{args.get('query', '')}»"
    if name == "web_search":
        return f"Веб-поиск: «{args.get('query', '')}» (за {args.get('days', 30)} дн.)"
    if name == "forecast_oil_price":
        return (f"Расчёт: прогноз {args.get('benchmark', 'brent').upper()} на {args.get('horizon_months', 3)} мес., "
                f"интервал {int(float(args.get('confidence', 0.8)) * 100)}%")
    if name == "supply_shock_scenario":
        return (f"Расчёт: сценарий изменения добычи на {args.get('supply_change_mbd')} млн б/с, "
                f"{args.get('horizon_months', 6)} мес.")
    return f"{name}({args})"

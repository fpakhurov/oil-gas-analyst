"""Перевод резюме рассуждений Claude на язык вопроса.

Резюме thinking-блоков (display="summarized") API формирует на английском, даже когда модель
отвечает и рассуждает по-русски. Для пользователя переводим их дешёвой быстрой моделью
(Claude Haiku 5.5, без thinking, effort=low). Перевод не блокирует ход: сегменты отправляются
в пул потоков по мере готовности, ошибка перевода оставляет оригинал.
"""

from __future__ import annotations

import logging
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from oilgas_analyst.config import Settings
from oilgas_analyst.observability import Usage, usage_from_message

log = logging.getLogger(__name__)

KEEP = ("Сохрани смысл, числа, единицы, тикеры, названия отчётов и организаций "
        "(EIA, OPEC, STEO, MOMR, Brent, WTI). Верни только перевод, без пояснений и кавычек.")


def _system_prompt(question: str) -> str:
    # Косвенное «на язык вопроса» Haiku иногда понимает как «оставь как есть» — называем язык явно
    if _cyrillic_share(question) >= 0.5:
        return ("Ты переводчик. Переведи текст пользователя на русский язык. Это краткое изложение хода "
                "рассуждений аналитика нефтегазового рынка. " + KEEP)
    return ("You are a translator. Translate the user's text into the language of this question: "
            f"«{question[:300]}». It is a short summary of an oil & gas analyst's reasoning. Keep numbers, "
            "units, tickers, report and organization names. Return only the translation.")


def _cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / max(len(letters), 1)


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("a" <= c.lower() <= "z" for c in letters) / max(len(letters), 1)


def needs_translation(question: str, text: str) -> bool:
    """Переводим, если вопрос не на латинице (русский и т.п.), а резюме — на латинице."""
    return bool(text.strip()) and _latin_share(question) < 0.5 and _latin_share(text) > 0.7


def _llm(s: Settings) -> ChatAnthropic:
    kwargs: dict = dict(model=s.aux_model, max_tokens=2000, reasoning_effort="low",
                        thinking={"type": "disabled"}, max_retries=2, timeout=60)
    if s.anthropic_api_key:
        kwargs["api_key"] = s.anthropic_api_key
    if s.anthropic_base_url:
        kwargs["base_url"] = s.anthropic_base_url
    return ChatAnthropic(**kwargs)


def translate_reasoning(question: str, text: str, s: Settings, store=None,
                        thread_id: str | None = None) -> tuple[str, Usage | None]:
    """Перевод одного резюме и его стоимость; при ошибке или если перевод не нужен — исходный текст."""
    if not (s.reasoning_translation and needs_translation(question, text)):
        return text, None
    if store is not None and s.daily_budget_usd > 0 and store.spent_today() >= s.daily_budget_usd:
        return text, None
    try:
        total = Usage()
        out = text
        for _ in range(2):  # повтор, если модель вернула текст непереведённым
            msg = _llm(s).invoke([SystemMessage(_system_prompt(question)), HumanMessage(text)])
            model, u = usage_from_message(msg, s.aux_model)
            total.add(u)
            if store is not None:
                store.record_usage(thread_id, model, u.input_tokens, u.output_tokens, u.cache_read,
                                   u.cache_write, u.cost_usd)
            out = re.sub(r"^\s*(Перевод|Translation)\s*:\s*", "", msg.text.strip(), flags=re.IGNORECASE) or text
            if not needs_translation(question, out):
                break
        return out, total
    except Exception as e:
        log.warning("Перевод резюме рассуждений не удался: %s", e)
        return text, None

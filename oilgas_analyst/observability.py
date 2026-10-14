"""Наблюдаемость: трейсинг в Langfuse (self-hosted) и расчёт стоимости вызовов Claude."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from oilgas_analyst.config import Settings

log = logging.getLogger(__name__)

# $ за 1M токенов: вход, выход, чтение кэша. Запись в кэш (TTL 5 мин) — 1.25 × цена входа.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
}
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    input_tokens: int = 0  # всего на входе, включая кэш
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    estimated: bool = False  # цена модели неизвестна — посчитано по ценам Opus

    def add(self, other: Usage) -> None:
        for f in ("input_tokens", "output_tokens", "cache_read", "cache_write", "cost_usd", "calls"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.estimated = self.estimated or other.estimated

    def to_dict(self) -> dict:
        return asdict(self)


def usage_from_message(msg, default_model: str) -> tuple[str, Usage]:
    """Usage и стоимость одного ответа модели (AIMessage с usage_metadata от langchain-anthropic)."""
    meta = getattr(msg, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    model = (getattr(msg, "response_metadata", None) or {}).get("model") or default_model
    u = Usage(
        input_tokens=int(meta.get("input_tokens") or 0),
        output_tokens=int(meta.get("output_tokens") or 0),
        cache_read=int(details.get("cache_read") or 0),
        cache_write=int(details.get("cache_creation") or 0),
        calls=1,
    )
    # Самое длинное совпадение: claude-opus-5-5 не должна тарифицироваться как claude-opus-5
    price = next((PRICES[n] for n in sorted(PRICES, key=len, reverse=True)
                  if model == n or model.startswith(n + "-")), None)
    if price is None:
        price, u.estimated = PRICES["claude-opus-5-5"], True
    p_in, p_out, p_cache = price
    uncached = max(u.input_tokens - u.cache_read - u.cache_write, 0)
    u.cost_usd = (uncached * p_in + u.cache_write * p_in * CACHE_WRITE_MULTIPLIER
                  + u.cache_read * p_cache + u.output_tokens * p_out) / 1_000_000
    return model, u


# --- Langfuse --------------------------------------------------------------

_client = None


def _langfuse(settings: Settings):
    """Один клиент Langfuse на процесс; ключи берём из настроек, а не из os.environ."""
    global _client
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                           base_url=settings.langfuse_base_url, environment="local")
    return _client


def tracing_callbacks(settings: Settings) -> list:
    """Колбэки LangChain для трейса одного хода; пустой список, если трейсинг выключен или недоступен."""
    if not settings.tracing_enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        _langfuse(settings)
        return [CallbackHandler(public_key=settings.langfuse_public_key)]
    except Exception as e:  # трейсинг не должен ломать ответ
        log.warning("Langfuse недоступен, трейсинг выключен: %s", e)
        return []


def trace_metadata(thread_id: str, model: str) -> dict:
    return {"langfuse_session_id": thread_id, "langfuse_tags": ["oilgas-analyst", model]}


def trace_url(settings: Settings, callbacks: list) -> str | None:
    trace_id = next((getattr(cb, "last_trace_id", None) for cb in callbacks if getattr(cb, "last_trace_id", None)), None)
    if not trace_id:
        return None
    # Ссылку собираем сами: SDK знает только внутренний адрес контейнера (langfuse-web:3000)
    return f"{settings.langfuse_ui_url.rstrip('/')}/project/{settings.langfuse_project_id}/traces/{trace_id}"


def flush(settings: Settings) -> None:
    """Отправка трейсов в фоне: недоступный Langfuse не должен задерживать ответ (при выходе
    из процесса SDK досылает остаток сам через atexit)."""
    if not (settings.tracing_enabled and _client is not None):
        return

    def _run():
        try:
            _client.flush()
        except Exception as e:
            log.warning("Не удалось отправить трейсы в Langfuse: %s", e)

    threading.Thread(target=_run, name="langfuse-flush", daemon=True).start()

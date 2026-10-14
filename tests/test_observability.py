"""Стоимость вызовов, трейсинг, хранилище диалогов/расходов, автообновление базы."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from oilgas_analyst import observability as obs
from oilgas_analyst import refresh
from oilgas_analyst.config import Settings
from oilgas_analyst.storage import get_store


@pytest.fixture
def settings(tmp_path):
    return Settings(anthropic_api_key="sk-ant-test", data_dir=tmp_path, langfuse_enabled=False)


def _msg(model, inp, out, read=0, write=0):
    return AIMessage("x", usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                                          "input_token_details": {"cache_read": read, "cache_creation": write}},
                     response_metadata={"model": model})


@pytest.mark.parametrize("model,expected", [
    # 1M некэшированных входных + 100k выходных
    ("claude-opus-5-5", 4.0 + 2.0),
    ("claude-sonnet-5-5", 2.0 + 1.0),
    ("claude-opus-5", 5.0 + 2.5),  # не путать с opus-5-5
])
def test_cost_by_model(model, expected):
    name, u = obs.usage_from_message(_msg(model, 1_000_000, 100_000), "claude-opus-5-5")
    assert name == model and u.cost_usd == pytest.approx(expected) and not u.estimated


def test_cache_pricing_and_unknown_model():
    # 1M входа: 600k из кэша ($0.20/M), 300k запись (×1.25 от $4), 100k обычных
    _, u = obs.usage_from_message(_msg("claude-opus-5-5", 1_000_000, 0, read=600_000, write=300_000), "x")
    assert u.cost_usd == pytest.approx(0.1 * 4 + 0.3 * 4 * 1.25 + 0.6 * 0.20)
    _, u = obs.usage_from_message(_msg("claude-future-9", 1_000_000, 0), "claude-future-9")
    assert u.estimated and u.cost_usd == pytest.approx(4.0)


def test_tracing_disabled_and_trace_url(settings):
    assert obs.tracing_callbacks(settings) == []
    s = settings.model_copy(update={"langfuse_ui_url": "http://localhost:3000/", "langfuse_project_id": "p1"})
    assert obs.trace_url(s, [SimpleNamespace(last_trace_id="abc")]) == "http://localhost:3000/project/p1/traces/abc"
    assert obs.trace_url(s, [SimpleNamespace(last_trace_id=None)]) is None
    assert obs.trace_metadata("t1", "m")["langfuse_session_id"] == "t1"


def test_tracing_requires_keys(tmp_path):
    s = Settings(data_dir=tmp_path, langfuse_enabled=True, langfuse_public_key="pk-lf-...")
    assert not s.tracing_enabled  # плейсхолдер из .env.example — ключа нет


def test_store_threads_turns_and_spend(settings):
    store = get_store(settings)
    store.upsert_thread("a", "Первый вопрос", "claude-opus-5-5")
    store.upsert_thread("a", "Другой заголовок не перетирает первый", "claude-opus-5-5")
    store.add_turn("a", "Первый вопрос", {"answer": "ответ", "usage": {"cost_usd": 0.1}})
    store.record_usage("a", "claude-opus-5-5", 10, 5, 0, 0, 0.1)
    store.record_usage("a", "claude-opus-5-5", 10, 5, 0, 0, 0.05)
    assert store.list_threads()[0]["title"] == "Первый вопрос"
    assert store.get_turns("a") == [{"question": "Первый вопрос", "answer": "ответ", "usage": {"cost_usd": 0.1}}]
    assert store.spent_today() == pytest.approx(0.15)
    store.delete_thread("a")
    assert store.list_threads() == [] and store.get_turns("a") == []


def test_refresh_schedule(settings, monkeypatch):
    assert refresh.is_due(settings)  # ещё ни разу не обновляли
    calls = []
    monkeypatch.setattr("oilgas_analyst.rag.download.download_reports", lambda **k: calls.append("dl") or {})
    monkeypatch.setattr("oilgas_analyst.rag.store.ingest_reports", lambda **k: calls.append("ingest") or {"indexed": []})
    monkeypatch.setattr("oilgas_analyst.forecasting.data.save_snapshot", lambda s: {})
    res = refresh.refresh_knowledge_base(settings)
    assert calls == ["dl", "ingest"] and "error" not in res
    assert not refresh.is_due(settings)
    later = datetime.now(timezone.utc) + timedelta(days=settings.refresh_interval_days, minutes=1)
    assert refresh.is_due(settings, now=later)
    assert not refresh.is_due(settings.model_copy(update={"refresh_interval_days": 0}))

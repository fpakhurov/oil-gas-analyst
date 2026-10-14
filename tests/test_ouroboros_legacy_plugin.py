"""Плагин Ouroboros: контракт get_tools() / handler(ctx, **args) -> str.

Если задан OUROBOROS_REPO — тест грузит настоящий ToolRegistry из форка Ouroboros,
иначе подставляет минимальную заглушку модуля ouroboros.tools.registry.
"""

import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "integrations" / "ouroboros" / "legacy_v6" / "oilgas.py"


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    # Плагин выставляет DATA_DIR в окружение — изолируем, чтобы не протекало в другие тесты
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "oilgas"))
    repo = os.environ.get("OUROBOROS_REPO")
    if repo:
        monkeypatch.syspath_prepend(repo)
    else:
        @dataclass
        class ToolContext:
            repo_dir: Path
            drive_root: Path
            current_chat_id: int | None = None

        @dataclass
        class ToolEntry:
            name: str
            schema: dict
            handler: object
            is_code_tool: bool = False
            timeout_sec: int = 120

        mod = types.ModuleType("ouroboros.tools.registry")
        mod.ToolContext, mod.ToolEntry = ToolContext, ToolEntry
        for name in ("ouroboros", "ouroboros.tools"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "ouroboros.tools.registry", mod)
    spec = importlib.util.spec_from_file_location("oilgas_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from ouroboros.tools.registry import ToolContext

    return module, ToolContext(repo_dir=tmp_path, drive_root=tmp_path)


def test_schemas(plugin):
    module, _ = plugin
    tools = module.get_tools()
    names = [t.name for t in tools]
    assert names[0] == "oilgas_ask" and len(names) == len(set(names)) == 7
    for t in tools:
        assert t.schema["name"] == t.name
        assert t.schema["parameters"]["type"] == "object"
        assert set(t.schema["parameters"]["required"]) <= set(t.schema["parameters"]["properties"])


def test_handler_returns_string_and_never_raises(plugin, monkeypatch):
    module, ctx = plugin
    import oilgas_analyst.forecasting.service as service

    monkeypatch.setattr(service, "run_forecast", lambda *a, **k: ("ПРОГНОЗ ok", {}, None))
    tools = {t.name: t for t in module.get_tools()}
    assert tools["oilgas_forecast"].handler(ctx, benchmark="brent", horizon_months=3) == "ПРОГНОЗ ok"

    def boom(*a, **k):
        raise RuntimeError("нет сети")

    monkeypatch.setattr(service, "run_forecast", boom)
    out = tools["oilgas_forecast"].handler(ctx)
    assert out.startswith("⚠️ oilgas_forecast") and "нет сети" in out


def test_add_report_validates_date(plugin):
    module, ctx = plugin
    tools = {t.name: t for t in module.get_tools()}
    out = tools["oilgas_add_report"].handler(ctx, url="http://x", publisher="OPEC", title="t",
                                             short_name="OPEC MOMR", date="March")
    assert "YYYY-MM" in out

"""Навык Ouroboros ≥ 7 (PluginAPI 2.0): регистрация, контракт обработчиков, маршруты виджета.

Фейковый PluginAPI повторяет используемую часть контракта ouroboros/contracts/plugin_api.py.
Полная проверка payload штатным skill_preflight Ouroboros описана в README.
"""

import asyncio
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "integrations" / "ouroboros" / "skill" / "plugin.py"


class FakeAPI:
    def __init__(self, state_dir: Path, settings: dict):
        self.state_dir, self._settings = state_dir, settings
        self.tools, self.routes, self.tabs, self.unload, self.logs = {}, {}, {}, [], []

    def register_tool(self, name, handler, *, description, schema, timeout_sec=60):
        assert re.fullmatch(r"[A-Za-z0-9_]{1,24}", name), name
        assert timeout_sec <= 300
        self.tools[name] = (handler, schema, description)

    def register_route(self, path, handler, *, methods=("GET",)):
        self.routes[path] = (handler, methods)

    def register_ui_tab(self, tab_id, title, *, icon=None, render=None):
        self.tabs[tab_id] = render

    def on_unload(self, fn):
        self.unload.append(fn)

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {k: v for k, v in self._settings.items() if k in keys}

    def log(self, level, msg):
        self.logs.append((level, msg))


class FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def skill(tmp_path, monkeypatch):
    for key in ("DATA_DIR", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)
    spec = importlib.util.spec_from_file_location("oilgas_skill_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = FakeAPI(tmp_path / "state", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    module.register(api)
    yield module, api
    for fn in api.unload:
        fn()
    from oilgas_analyst.config import get_settings

    get_settings.cache_clear()


def test_registration(skill):
    _, api = skill
    assert set(api.tools) == {"ask", "search_reports", "web_search", "forecast", "supply_shock",
                              "refresh_reports", "add_report"}
    for _, schema, desc in api.tools.values():
        assert schema["type"] == "object" and desc
        assert set(schema["required"]) <= set(schema["properties"])
    assert set(api.routes) == {"ask", "forecast"} and all(m == ("POST",) for _, m in api.routes.values())
    routes_in_ui = {c["route"] for c in api.tabs["analyst"]["components"] if c["type"] == "form"}
    assert routes_in_ui <= set(api.routes)


def test_state_dir_and_granted_key(skill, monkeypatch):
    module, api = skill
    import oilgas_analyst.forecasting.service as service

    monkeypatch.setattr(service, "run_forecast", lambda *a, **k: ("ПРОГНОЗ ok", {}, None))
    assert api.tools["forecast"][0](benchmark="brent", horizon_months=3) == "ПРОГНОЗ ok"
    assert os.environ["DATA_DIR"] == str(api.state_dir / "data")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
    from oilgas_analyst.config import get_settings

    assert get_settings().reports_dir == api.state_dir / "data" / "reports"


def test_errors_are_json_not_exceptions(skill, monkeypatch):
    _, api = skill
    import oilgas_analyst.forecasting.service as service

    def boom(*a, **k):
        raise RuntimeError("нет сети")

    monkeypatch.setattr(service, "run_forecast", boom)
    out = json.loads(api.tools["forecast"][0]())
    assert "нет сети" in out["error"]
    assert "error" in json.loads(api.tools["ask"][0](question=""))
    assert "YYYY-MM" in json.loads(api.tools["add_report"][0](url="https://x/y.pdf", short_name="OPEC MOMR",
                                                               date="March"))["error"]
    assert "http" in json.loads(api.tools["add_report"][0](url="file:///etc/passwd", short_name="x",
                                                           date="2026-09"))["error"]


def test_forecast_route_dispatches_scenario(skill, monkeypatch):
    module, api = skill
    calls = []
    monkeypatch.setattr(module, "_tool_supply_shock", lambda **k: calls.append(("shock", k)) or "S")
    monkeypatch.setattr(module, "_tool_forecast", lambda **k: calls.append(("fc", k)) or "F")
    handler = api.routes["forecast"][0]
    asyncio.run(handler(FakeRequest({"benchmark": "WTI", "horizon_months": "6", "supply_change_mbd": "-1,5"})))
    asyncio.run(handler(FakeRequest({"benchmark": "brent"})))
    assert calls[0] == ("shock", {"supply_change_mbd": -1.5, "benchmark": "wti", "horizon_months": 6})
    assert calls[1] == ("fc", {"benchmark": "brent", "horizon_months": 3})
    resp = asyncio.run(handler(FakeRequest({"benchmark": "urals"})))
    assert "brent или wti" in json.loads(resp.body)["report"]

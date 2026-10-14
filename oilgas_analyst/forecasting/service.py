"""Фасад расчётного модуля: загрузка данных → модель → текстовая интерпретация + данные для графика."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.forecasting.data import BENCHMARKS, load_prices
from oilgas_analyst.forecasting.models import ForecastResult, forecast
from oilgas_analyst.forecasting.scenario import ScenarioResult, supply_shock

RU_MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def _m(ts: pd.Timestamp) -> str:
    return f"{RU_MONTHS_SHORT[ts.month - 1]} {ts.year}"


def _history_payload(monthly: pd.Series, months: int = 36) -> dict:
    tail = monthly.iloc[-months:]
    return {"dates": [d.strftime("%Y-%m-%d") for d in tail.index], "values": [round(float(v), 2) for v in tail]}


def _series_payload(s: pd.Series) -> list[float]:
    return [round(float(v), 2) for v in s.values]


def _trend_word(change_pct: float) -> str:
    if change_pct > 3:
        return "умеренный рост" if change_pct < 10 else "заметный рост"
    if change_pct < -3:
        return "умеренное снижение" if change_pct > -10 else "заметное снижение"
    return "боковую динамику"


def run_forecast(benchmark: str = "brent", horizon_months: int = 3, confidence: float = 0.8,
                 method: str = "ensemble", settings: Settings | None = None) -> tuple[str, dict, ForecastResult]:
    s = settings or get_settings()
    hist = load_prices(benchmark, s)
    methods = ("sarima", "ets") if method in ("ensemble", "all") else (method,)
    res = forecast(hist.monthly, horizon_months, confidence, methods, benchmark)
    name = BENCHMARKS[benchmark]["name"]
    ens = res.ensemble
    last_m = hist.monthly.iloc[-1]
    end_mean = float(ens.mean.iloc[-1])
    change = (end_mean / last_m - 1) * 100
    width = (float(ens.upper.iloc[-1]) - float(ens.lower.iloc[-1])) / end_mean * 100

    lines = [
        f"ПРОГНОЗ {name}, горизонт {horizon_months} мес., доверительный интервал {int(confidence*100)}%",
        f"Данные: {hist.source}; последняя цена {hist.last_price:.2f} $/барр. на {hist.last_date:%d.%m.%Y}; "
        f"последний полный месяц ({_m(hist.monthly.index[-1])}) — в среднем {last_m:.2f} $/барр.",
        "",
        "Месяц | Прогноз (ансамбль) | Интервал",
    ]
    for ts, mu, lo, hi in zip(ens.mean.index, ens.mean, ens.lower, ens.upper):
        lines.append(f"{_m(ts)} | {mu:.1f} | {lo:.1f} – {hi:.1f}")
    lines += ["", "Методы:"]
    for mf in res.methods:
        mape = f"MAPE бэктеста {mf.backtest_mape*100:.1f}%" if mf.backtest_mape is not None and np.isfinite(mf.backtest_mape) else "бэктест недоступен"
        lines.append(f"- {mf.method}: {mf.details}; {mape}; вес в ансамбле {mf.weight:.2f}; "
                     f"на конец горизонта {mf.mean.iloc[-1]:.1f} ({mf.lower.iloc[-1]:.1f} – {mf.upper.iloc[-1]:.1f})")
    lines += [
        "",
        f"Интерпретация: модели указывают на {_trend_word(change)} ({change:+.1f}% к среднему за последний месяц) "
        f"к {_m(ens.mean.index[-1])}. Ширина интервала на конец горизонта ≈ {width:.0f}% от прогноза — "
        f"{'высокая' if width > 30 else 'умеренная'} неопределённость.",
        "Ограничения: модели чисто статистические (экстраполяция ценового ряда), не учитывают решения ОПЕК+, "
        "санкции, геополитику и баланс спроса/предложения — эти факторы нужно оценивать отдельно "
        "(отчёты агентств, новости, сценарный модуль supply_shock_scenario).",
    ]
    payload = {
        "kind": "forecast",
        "benchmark": name,
        "confidence": confidence,
        "source": hist.source,
        "history": _history_payload(hist.monthly),
        "forecast": {
            "dates": [d.strftime("%Y-%m-%d") for d in ens.mean.index],
            "mean": _series_payload(ens.mean),
            "lower": _series_payload(ens.lower),
            "upper": _series_payload(ens.upper),
            "methods": {m.method: _series_payload(m.mean) for m in res.methods},
        },
    }
    return "\n".join(lines), payload, res


def run_supply_shock(supply_change_mbd: float, benchmark: str = "brent", horizon_months: int = 6,
                     offset_share: float = 0.0, phase_in_months: int = 2, confidence: float = 0.8,
                     settings: Settings | None = None) -> tuple[str, dict, ScenarioResult]:
    s = settings or get_settings()
    _, _, base = run_forecast(benchmark, horizon_months, confidence, "ensemble", s)
    sc = supply_shock(base, supply_change_mbd, s.global_supply_mbd, offset_share, phase_in_months)
    name = BENCHMARKS[benchmark]["name"]
    direction = "сокращение" if supply_change_mbd < 0 else "увеличение"
    end = sc.baseline.index[-1]

    lines = [
        f"СЦЕНАРИЙ: {direction} предложения на {abs(supply_change_mbd):.2f} млн б/с, {name}, горизонт {horizon_months} мес.",
        f"Эффективный шок с учётом компенсации другими производителями ({offset_share:.0%}): "
        f"{sc.effective_shock_mbd:+.2f} млн б/с = {sc.shock_share*100:+.2f}% мирового предложения "
        f"({s.global_supply_mbd:.0f} млн б/с).",
        "",
        "Оценка ценового эффекта (полный эффект после выхода на плато):",
    ]
    for k, v in sc.price_impact_pct.items():
        lines.append(f"- {k} (сумма эластичностей {sc.assumptions['elasticity_sum'][k]}): {v:+.1f}% → "
                     f"{sc.paths[k].iloc[-1]:.1f} $/барр. в {_m(end)}")
    lines += [
        f"- базовый прогноз без шока: {sc.baseline.iloc[-1]:.1f} $/барр. в {_m(end)}",
        f"- общий диапазон с учётом статистической неопределённости ({int(confidence*100)}%): "
        f"{sc.lower.iloc[-1]:.1f} – {sc.upper.iloc[-1]:.1f} $/барр.",
        "",
        "Помесячно (база → центральный сценарий, диапазон):",
    ]
    for ts, b, c, lo, hi in zip(sc.baseline.index, sc.baseline, sc.paths["центральный"], sc.lower, sc.upper):
        lines.append(f"{_m(ts)}: {b:.1f} → {c:.1f} ({lo:.1f} – {hi:.1f})")
    lines += [
        "",
        f"Допущения: эффект входит линейно за {phase_in_months} мес.; эластичности — краткосрочные оценки из "
        "литературы (Caldara et al. 2019; Baumeister & Hamilton 2019; Kilian & Murphy 2014). "
        "Модель не учитывает реакцию запасов, изменение резервных мощностей ОПЕК+ и ожидания рынка "
        "(часть эффекта может быть уже в цене, если решение было ожидаемым).",
    ]
    payload = {
        "kind": "scenario",
        "benchmark": name,
        "confidence": confidence,
        "history": _history_payload(load_prices(benchmark, s).monthly),
        "forecast": {
            "dates": [d.strftime("%Y-%m-%d") for d in sc.baseline.index],
            "mean": _series_payload(sc.paths["центральный"]),
            "lower": _series_payload(sc.lower),
            "upper": _series_payload(sc.upper),
            "methods": {"База (без шока)": _series_payload(sc.baseline),
                        **{f"Сценарий: {k}": _series_payload(v) for k, v in sc.paths.items() if k != "центральный"}},
        },
    }
    return "\n".join(lines), payload, sc

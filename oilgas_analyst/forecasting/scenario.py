"""Сценарный модуль: оценка влияния изменения добычи на цену нефти.

Модель частичного равновесия с постоянными эластичностями:

    Δln(P) ≈ −Δln(Q_supply_shock) / (ε_s + |ε_d|)

где шок предложения ΔQ (млн б/с) выражен в доле мирового предложения, а ε_s и ε_d —
краткосрочные эластичности предложения и спроса по цене. Сумма эластичностей
задаётся диапазоном по литературе:
  • Caldara, Cavallo, Iacoviello (2019): ε_s ≈ 0.1, ε_d ≈ −0.1 → сумма ≈ 0.2 (центральный);
  • Baumeister & Hamilton (2019): ε_d ≈ −0.35, ε_s ≈ 0.15 → сумма ≈ 0.5 (мягкий сценарий);
  • Kilian & Murphy (2014) и оценки с почти неэластичным спросом → сумма ≈ 0.1 (жёсткий).

Часть шока может компенсироваться другими производителями (резервные мощности,
сланец, запасы) — параметр offset_share. Эффект накладывается на базовый
статистический прогноз (ансамбль SARIMA+ETS) с постепенным входом (phase_in_months).
Это стилизованная оценка порядка величины, а не структурная модель рынка.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oilgas_analyst.forecasting.models import ForecastResult

ELASTICITY_SUM = {"мягкий": 0.5, "центральный": 0.2, "жёсткий": 0.1}


@dataclass
class ScenarioResult:
    supply_change_mbd: float
    effective_shock_mbd: float
    shock_share: float
    price_impact_pct: dict[str, float]  # по каждому сценарию эластичности
    baseline: pd.Series
    paths: dict[str, pd.Series]  # цена по сценариям
    lower: pd.Series  # нижняя граница: мин(базовый CI-low, сценарии) с учётом эффекта
    upper: pd.Series
    assumptions: dict


def supply_shock(base: ForecastResult, supply_change_mbd: float, global_supply_mbd: float = 104.0,
                 offset_share: float = 0.0, phase_in_months: int = 1) -> ScenarioResult:
    """supply_change_mbd < 0 — сокращение добычи (цена растёт), > 0 — рост добычи."""
    if not -20 <= supply_change_mbd <= 20:
        raise ValueError("Изменение добычи должно быть в пределах ±20 млн б/с")
    if not 0 <= offset_share <= 1:
        raise ValueError("offset_share должен быть в [0, 1]")

    effective = supply_change_mbd * (1 - offset_share)
    share = effective / global_supply_mbd
    h = base.horizon
    ramp = np.minimum(np.arange(1, h + 1) / max(phase_in_months, 1), 1.0)

    baseline = base.ensemble.mean
    impacts, paths = {}, {}
    for name, eps in ELASTICITY_SUM.items():
        log_impact = -np.log1p(share) / eps  # сокращение (share<0) → рост цены
        impacts[name] = float(np.expm1(log_impact) * 100)
        paths[name] = baseline * np.exp(log_impact * ramp)

    # Неопределённость = статистический интервал базы × диапазон эластичностей
    soft, hard = paths["мягкий"] / baseline, paths["жёсткий"] / baseline
    lo_mult, hi_mult = np.minimum(soft, hard), np.maximum(soft, hard)
    lower = base.ensemble.lower * lo_mult
    upper = base.ensemble.upper * hi_mult

    return ScenarioResult(
        supply_change_mbd=supply_change_mbd,
        effective_shock_mbd=effective,
        shock_share=share,
        price_impact_pct=impacts,
        baseline=baseline,
        paths=paths,
        lower=lower,
        upper=upper,
        assumptions={
            "global_supply_mbd": global_supply_mbd,
            "offset_share": offset_share,
            "phase_in_months": phase_in_months,
            "elasticity_sum": ELASTICITY_SUM,
        },
    )

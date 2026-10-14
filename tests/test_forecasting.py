import numpy as np
import pandas as pd
import pytest

from oilgas_analyst.forecasting.models import forecast
from oilgas_analyst.forecasting.scenario import supply_shock


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(0)
    n = 120
    log_p = np.log(70) + np.cumsum(rng.normal(0, 0.06, n)) + 0.05 * np.sin(np.arange(n) * 2 * np.pi / 12)
    return pd.Series(np.exp(log_p), index=pd.date_range("2016-01-01", periods=n, freq="MS"))


@pytest.fixture(scope="module")
def result(monthly):
    return forecast(monthly, horizon=3, confidence=0.8)


def test_forecast_shape_and_interval(result):
    ens = result.ensemble
    assert len(ens.mean) == 3
    assert (ens.lower.values < ens.mean.values).all() and (ens.mean.values < ens.upper.values).all()
    assert {m.method for m in result.methods} == {"SARIMA", "ETS"}


def test_ensemble_weights_sum_to_one(result):
    assert sum(m.weight for m in result.methods) == pytest.approx(1.0)
    assert all(m.backtest_mape is not None for m in result.methods)


def test_interval_widens_with_horizon(result):
    width = result.ensemble.upper.values - result.ensemble.lower.values
    assert width[-1] > width[0]


def test_higher_confidence_wider(monthly):
    narrow = forecast(monthly, 3, 0.8, backtest=False).ensemble
    wide = forecast(monthly, 3, 0.95, backtest=False).ensemble
    assert (wide.upper.values - wide.lower.values > narrow.upper.values - narrow.lower.values).all()


def test_validation(monthly):
    with pytest.raises(ValueError):
        forecast(monthly, horizon=0)
    with pytest.raises(ValueError):
        forecast(monthly, horizon=3, confidence=1.0)


def test_supply_cut_raises_price(result):
    sc = supply_shock(result, -2.0, global_supply_mbd=100, phase_in_months=1)
    assert all(v > 0 for v in sc.price_impact_pct.values())
    # чем ниже суммарная эластичность, тем сильнее эффект
    assert sc.price_impact_pct["жёсткий"] > sc.price_impact_pct["центральный"] > sc.price_impact_pct["мягкий"]
    # 2% шок при сумме эластичностей 0.2 ≈ +10.6% (ln(1/0.98)/0.2)
    assert sc.price_impact_pct["центральный"] == pytest.approx((np.exp(-np.log(0.98) / 0.2) - 1) * 100, rel=1e-6)


def test_supply_increase_lowers_price_and_offset_dampens(result):
    up = supply_shock(result, +1.0)
    assert all(v < 0 for v in up.price_impact_pct.values())
    full = supply_shock(result, -1.0, offset_share=0.0)
    half = supply_shock(result, -1.0, offset_share=0.5)
    assert half.price_impact_pct["центральный"] < full.price_impact_pct["центральный"]


def test_phase_in(result):
    sc = supply_shock(result, -1.0, phase_in_months=3)
    ratio = (sc.paths["центральный"] / sc.baseline).values
    assert ratio[0] < ratio[1] < ratio[2]


def test_current_month_is_excluded_from_monthly_series():
    from datetime import date

    from oilgas_analyst.forecasting.data import _to_monthly

    days = pd.bdate_range("2026-08-01", "2026-10-14")
    daily = pd.Series(range(len(days)), index=days, dtype=float)
    assert _to_monthly(daily, today=date(2026, 10, 14)).index[-1] == pd.Timestamp("2026-09-01")
    # исторический ряд (CSV-снимок) — последний месяц не текущий, остаётся
    assert _to_monthly(daily, today=date(2026, 12, 1)).index[-1] == pd.Timestamp("2026-10-01")

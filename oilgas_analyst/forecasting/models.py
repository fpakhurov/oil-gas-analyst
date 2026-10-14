"""Статистические модели прогноза: SARIMA и экспоненциальное сглаживание (ETS) + ансамбль.

Обе модели строятся на логарифме среднемесячной цены: так доверительные интервалы
получаются асимметричными (цена не уходит ниже нуля) и ближе к реальной динамике.
Веса ансамбля — обратно пропорциональны ошибке (MAPE) на скользящем бэктесте.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MethodForecast:
    method: str
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    details: str = ""
    backtest_mape: float | None = None
    weight: float | None = None
    spec: tuple | None = None


@dataclass
class ForecastResult:
    benchmark: str
    horizon: int
    confidence: float
    ensemble: MethodForecast
    methods: list[MethodForecast] = field(default_factory=list)


def _sarima(log_y: pd.Series, h: int, alpha: float, spec: tuple | None = None) -> MethodForecast:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best = None
    # Небольшой перебор порядков по AIC: d=1 (цены нестационарны), сезонность по году.
    # В бэктесте spec фиксирован (выбран на полной выборке), чтобы не перебирать заново.
    grid = [spec] if spec else [((p, 1, q), seasonal) for p, q in itertools.product(range(3), range(3))
                                for seasonal in [(0, 0, 0, 0), (1, 0, 0, 12)]]
    for order, seasonal in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(log_y, order=order, seasonal_order=seasonal,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            if best is None or res.aic < best[0]:
                best = (res.aic, order, seasonal, res)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("SARIMA: ни одна спецификация не сошлась")
    aic, order, seasonal, res = best
    frame = res.get_forecast(h).summary_frame(alpha=alpha)
    sname = f"x{seasonal}" if seasonal[3] else ""
    return MethodForecast(
        "SARIMA", np.exp(frame["mean"]), np.exp(frame["mean_ci_lower"]), np.exp(frame["mean_ci_upper"]),
        details=f"SARIMA{order}{sname}, AIC={aic:.1f}", spec=(order, seasonal),
    )


def _ets(log_y: pd.Series, h: int, alpha: float, spec: tuple | None = None) -> MethodForecast:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ETSModel(log_y, error="add", trend="add", damped_trend=True).fit(disp=False)
    pred = res.get_prediction(start=len(log_y), end=len(log_y) + h - 1)
    frame = pred.summary_frame(alpha=alpha)
    return MethodForecast(
        "ETS", np.exp(frame["mean"]), np.exp(frame["pi_lower"]), np.exp(frame["pi_upper"]),
        details=f"ETS(A,Ad,N) с затухающим трендом, phi={dict(zip(res.param_names, res.params)).get('damping_trend', float('nan')):.2f}",
    )


METHODS = {"sarima": _sarima, "ets": _ets}


def _backtest_mape(monthly: pd.Series, fn, h: int, spec: tuple | None = None, folds: int = 4) -> float:
    """Скользящий бэктест: прогноз на h месяцев из нескольких точек в прошлом."""
    errors = []
    log_y = np.log(monthly)
    for k in range(folds, 0, -1):
        cut = len(log_y) - k * h
        if cut < 36:
            continue
        train, test = log_y.iloc[:cut], monthly.iloc[cut : cut + h]
        try:
            fc = fn(train, len(test), 0.2, spec)
            errors.append(float(np.mean(np.abs(fc.mean.values - test.values) / test.values)))
        except Exception as e:
            log.debug("backtest fold failed: %s", e)
    return float(np.mean(errors)) if errors else float("nan")


def forecast(monthly: pd.Series, horizon: int = 3, confidence: float = 0.8,
             methods: tuple[str, ...] = ("sarima", "ets"), benchmark: str = "brent",
             backtest: bool = True) -> ForecastResult:
    if not 1 <= horizon <= 24:
        raise ValueError("Горизонт прогноза должен быть от 1 до 24 месяцев")
    if not 0.5 <= confidence < 1:
        raise ValueError("Уровень доверия должен быть в диапазоне [0.5, 1)")
    alpha = 1 - confidence
    log_y = np.log(monthly.astype(float))

    results = []
    for name in methods:
        fn = METHODS[name]
        try:
            mf = fn(log_y, horizon, alpha)
        except Exception as e:
            log.warning("Метод %s не отработал: %s", name, e)
            continue
        if backtest:
            mf.backtest_mape = _backtest_mape(monthly, fn, max(horizon, 3), mf.spec)
        results.append(mf)
    if not results:
        raise RuntimeError("Ни один метод прогноза не отработал")

    # Веса ансамбля: 1/MAPE; если бэктест недоступен — поровну
    mapes = np.array([m.backtest_mape if m.backtest_mape and np.isfinite(m.backtest_mape) else np.nan for m in results])
    if np.all(np.isfinite(mapes)):
        w = (1 / mapes) / np.sum(1 / mapes)
    else:
        w = np.full(len(results), 1 / len(results))
    for m, wi in zip(results, w):
        m.weight = float(wi)

    idx = results[0].mean.index
    ens_mean = sum(m.mean.values * m.weight for m in results)
    # Интервал ансамбля — консервативный: огибающая интервалов отдельных моделей
    ens_lower = np.min([m.lower.values for m in results], axis=0)
    ens_upper = np.max([m.upper.values for m in results], axis=0)
    ensemble = MethodForecast(
        "Ансамбль", pd.Series(ens_mean, idx), pd.Series(ens_lower, idx), pd.Series(ens_upper, idx),
        details="взвешенное среднее (веса 1/MAPE бэктеста), интервал — огибающая интервалов моделей",
    )
    return ForecastResult(benchmark, horizon, confidence, ensemble, results)

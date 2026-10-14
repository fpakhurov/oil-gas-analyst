"""Загрузка исторических цен на нефть.

Цепочка источников (первый успешный выигрывает):
1. yfinance — фьючерсы ICE Brent (BZ=F) / NYMEX WTI (CL=F), дневные цены закрытия;
2. FRED (EIA spot, DCOILBRENTEU / DCOILWTICO) — CSV без ключа;
3. Подготовленный CSV в data/prices/ — офлайн-резерв, кладётся скриптом scripts/download_prices.py.

Модели работают на среднемесячных ценах: дневной шум для горизонта 1–12 мес. не нужен.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from oilgas_analyst.config import Settings, get_settings

log = logging.getLogger(__name__)

BENCHMARKS = {
    "brent": {"yahoo": "BZ=F", "fred": "DCOILBRENTEU", "name": "Brent"},
    "wti": {"yahoo": "CL=F", "fred": "DCOILWTICO", "name": "WTI"},
}


@dataclass
class PriceHistory:
    benchmark: str
    daily: pd.Series  # USD/bbl, индекс — дата
    monthly: pd.Series  # среднемесячные, индекс — начало месяца
    source: str

    @property
    def last_price(self) -> float:
        return float(self.daily.iloc[-1])

    @property
    def last_date(self) -> date:
        return self.daily.index[-1].date()


def _from_yahoo(ticker: str, start: date) -> pd.Series:
    import yfinance as yf

    # Yahoo периодически отдаёт пустой ответ («possibly delisted») — повторяем, прежде чем уйти
    # на резервный источник: FRED даёт спот, а не фьючерс, и прогноз стал бы несопоставим
    for attempt in range(3):
        df = yf.download(ticker, start=start.isoformat(), progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise ValueError(f"yfinance вернул пустые данные для {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # новые версии yfinance отдают MultiIndex-колонки
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def _from_fred(series_id: str, start: date) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start.isoformat()}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    if s.empty:
        raise ValueError(f"FRED вернул пустой ряд {series_id}")
    return s


def _from_csv(benchmark: str, settings: Settings) -> pd.Series:
    path = settings.prices_dir / f"{benchmark}_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["price"].astype(float)


def _to_monthly(daily: pd.Series, today: date | None = None) -> pd.Series:
    monthly = daily.resample("MS").mean().dropna()
    # Текущий месяц ещё идёт: его среднее по части дней смещает ряд — модели строим по полным месяцам
    today = today or date.today()
    if daily.index[-1].to_period("M") == pd.Period(today, "M"):
        monthly = monthly.iloc[:-1]
    monthly.index.freq = "MS"
    return monthly


def save_snapshot(settings: Settings | None = None) -> dict[str, str]:
    """Сохраняет дневные цены в data/prices/*.csv — офлайн-резерв для load_prices."""
    s = settings or get_settings()
    s.prices_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for bm in BENCHMARKS:
        hist = load_prices(bm, s)
        path = s.prices_dir / f"{bm}_daily.csv"
        hist.daily.rename("price").rename_axis("date").round(3).to_csv(path)
        out[bm] = f"{path.name}: {len(hist.daily)} строк, {hist.source}, до {hist.last_date}"
    return out


_cache: dict[tuple[str, str], PriceHistory] = {}


def load_prices(benchmark: str = "brent", settings: Settings | None = None) -> PriceHistory:
    s = settings or get_settings()
    benchmark = benchmark.lower()
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Неизвестный бенчмарк {benchmark!r}. Доступны: {', '.join(BENCHMARKS)}")
    key = (benchmark, datetime.now().strftime("%Y-%m-%d"))
    if key in _cache:
        return _cache[key]

    cfg = BENCHMARKS[benchmark]
    start = date.today() - timedelta(days=365 * s.price_history_years)
    attempts = [
        (f"Yahoo Finance ({cfg['yahoo']}, фьючерс)", lambda: _from_yahoo(cfg["yahoo"], start)),
        (f"FRED/EIA ({cfg['fred']}, спот)", lambda: _from_fred(cfg["fred"], start)),
        (f"локальный CSV data/prices/{benchmark}_daily.csv", lambda: _from_csv(benchmark, s)),
    ]
    errors = []
    for name, fn in attempts:
        try:
            daily = fn()
            daily = daily[daily > 0].sort_index()
            hist = PriceHistory(benchmark, daily, _to_monthly(daily), name)
            _cache[key] = hist
            return hist
        except Exception as e:
            log.warning("Источник цен %s недоступен: %s", name, e)
            errors.append(f"{name}: {e}")
    raise RuntimeError("Не удалось загрузить цены ни из одного источника:\n" + "\n".join(errors))

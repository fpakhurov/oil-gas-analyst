"""Загрузка PDF-отчётов в data/reports.

* EIA Short-Term Energy Outlook — публичный домен США, скачивается напрямую
  (текущий выпуск + архив за предыдущие месяцы).
* OPEC MOMR и IEA OMR закрыты Cloudflare/подпиской от автоматического скачивания:
  положите PDF вручную в data/reports и при необходимости опишите их в catalog.json
  (или назовите файл по шаблону `opec_momr_2026-09.pdf` / `iea_omr_2026-09.pdf`).
  Для OPEC пробуем известные прямые ссылки из EXTRA_REPORTS (best effort).
"""

from __future__ import annotations

import logging
from datetime import date

import requests

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.rag.catalog import ReportMeta, load_catalog, save_catalog

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (oilgas-analyst; research use)"}
MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Дополнительные отчёты с прямыми ссылками: (имя файла, url, метаданные)
EXTRA_REPORTS: list[tuple[str, str, dict]] = [
    ("opec_momr_2025-10.pdf", "https://static.poder360.com.br/2025/10/relatorio-opep-outubro.pdf",
     {"publisher": "OPEC", "title": "Monthly Oil Market Report (MOMR)", "short_name": "OPEC MOMR", "date": "2025-10"}),
]


def _months_back(n: int, today: date | None = None) -> list[tuple[int, int]]:
    d = today or date.today()
    y, m = d.year, d.month
    out = []
    for _ in range(n):
        out.append((y, m))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return out


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


def _first_page_mentions(path, text: str) -> bool:
    from pypdf import PdfReader

    try:
        return text.lower() in (PdfReader(str(path)).pages[0].extract_text() or "").lower()
    except Exception:
        return False


def _fetch(url: str, dest) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=120)
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            log.warning("Не PDF или ошибка %s: %s", r.status_code, url)
            return False
        dest.write_bytes(r.content)
        return True
    except requests.RequestException as e:
        log.warning("Не удалось скачать %s: %s", url, e)
        return False


def download_reports(months: int = 3, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(s.reports_dir)
    result = {"downloaded": [], "existing": [], "failed": []}

    # EIA STEO: архивные выпуски лежат по адресу archives/<mon><yy>.pdf; текущий — steo_full.pdf
    for y, m in _months_back(months + 1):
        fname = f"eia_steo_{y}-{m:02d}.pdf"
        dest = s.reports_dir / fname
        if dest.exists():
            result["existing"].append(fname)
            continue
        url = f"https://www.eia.gov/outlooks/steo/archives/{MONTH_ABBR[m-1]}{y % 100:02d}.pdf"
        ok = _fetch(url, dest)
        if not ok and (y, m) == (date.today().year, date.today().month):
            # Текущий выпуск попадает в архив только в следующем месяце
            url = "https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf"
            ok = _fetch(url, dest) and _first_page_mentions(dest, f"{MONTH_NAMES[m-1]} {y}")
            if not ok:
                dest.unlink(missing_ok=True)
        if ok:
            catalog[fname] = ReportMeta(fname, "EIA", "Short-Term Energy Outlook (STEO)", "EIA STEO", f"{y}-{m:02d}", url)
            result["downloaded"].append(fname)
        if len([f for f in result["downloaded"] + result["existing"] if f.startswith("eia_steo")]) >= months:
            break

    for fname, url, meta in EXTRA_REPORTS:
        dest = s.reports_dir / fname
        if dest.exists():
            result["existing"].append(fname)
        elif _fetch(url, dest):
            catalog[fname] = ReportMeta(fname, url=url, **meta)
            result["downloaded"].append(fname)
        else:
            result["failed"].append(fname)

    save_catalog(s.reports_dir, catalog)
    return result

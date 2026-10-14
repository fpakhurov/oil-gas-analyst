"""Каталог отчётов: метаданные (издатель, название, дата) для PDF в data/reports.

Метаданные берутся из data/reports/catalog.json; если файла там нет —
пытаемся угадать по имени файла (например, `eia_steo_2026-10.pdf`).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

RU_MONTHS = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь",
    7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
}

KNOWN_PUBLISHERS = {
    "opec": ("OPEC", "Monthly Oil Market Report (MOMR)", "OPEC MOMR"),
    "momr": ("OPEC", "Monthly Oil Market Report (MOMR)", "OPEC MOMR"),
    "eia_steo": ("EIA", "Short-Term Energy Outlook (STEO)", "EIA STEO"),
    "steo": ("EIA", "Short-Term Energy Outlook (STEO)", "EIA STEO"),
    "iea_omr": ("IEA", "Oil Market Report (OMR)", "IEA OMR"),
    "iea": ("IEA", "Oil Market Report (OMR)", "IEA OMR"),
}


@dataclass
class ReportMeta:
    file: str
    publisher: str
    title: str
    short_name: str
    date: str  # YYYY-MM or YYYY-MM-DD
    url: str = ""

    @property
    def human_date(self) -> str:
        m = re.match(r"(\d{4})-(\d{2})", self.date or "")
        if not m:
            return self.date or "дата не указана"
        return f"{RU_MONTHS[int(m.group(2))]} {m.group(1)}"

    def label(self, page: int | None = None) -> str:
        """Метка источника для ответа: [Отчёт EIA STEO, октябрь 2026, стр. 5]."""
        page_part = f", стр. {page}" if page else ""
        return f"[Отчёт {self.short_name}, {self.human_date}{page_part}]"

    def to_dict(self) -> dict:
        return asdict(self)


def load_catalog(reports_dir: Path) -> dict[str, ReportMeta]:
    path = reports_dir / "catalog.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["file"]: ReportMeta(**item) for item in raw.get("reports", [])}


def save_catalog(reports_dir: Path, catalog: dict[str, ReportMeta]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    items = sorted((m.to_dict() for m in catalog.values()), key=lambda d: (d["publisher"], d["date"]))
    (reports_dir / "catalog.json").write_text(
        json.dumps({"reports": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def guess_meta(file_name: str) -> ReportMeta:
    stem = Path(file_name).stem.lower()
    date = ""
    if m := re.search(r"(20\d{2})[-_.]?(0[1-9]|1[0-2])", stem):
        date = f"{m.group(1)}-{m.group(2)}"
    for key, (publisher, title, short) in KNOWN_PUBLISHERS.items():
        if key in stem:
            return ReportMeta(file_name, publisher, title, short, date)
    name = Path(file_name).stem
    return ReportMeta(file_name, "Unknown", name, name, date)


def meta_for(file_name: str, catalog: dict[str, ReportMeta]) -> ReportMeta:
    return catalog.get(file_name) or guess_meta(file_name)

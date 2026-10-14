from pathlib import Path

import pytest

from oilgas_analyst.config import Settings
from oilgas_analyst.rag.catalog import ReportMeta, guess_meta
from oilgas_analyst.rag.store import RetrievedChunk, chunk_report, format_chunks

REPORTS = Path(__file__).resolve().parent.parent / "data" / "reports"


def test_label_format():
    m = ReportMeta("x.pdf", "OPEC", "Monthly Oil Market Report (MOMR)", "OPEC MOMR", "2025-03")
    assert m.label(12) == "[Отчёт OPEC MOMR, март 2025, стр. 12]"
    assert m.label() == "[Отчёт OPEC MOMR, март 2025]"


@pytest.mark.parametrize("name,short,date", [
    ("eia_steo_2026-10.pdf", "EIA STEO", "2026-10"),
    ("opec_momr_2025_09.pdf", "OPEC MOMR", "2025-09"),
    ("IEA_OMR_202607.pdf", "IEA OMR", "2026-07"),
    ("random_report.pdf", "random_report", ""),
])
def test_guess_meta(name, short, date):
    m = guess_meta(name)
    assert m.short_name == short and m.date == date


def test_format_chunks_marks_relevance():
    chunks = [RetrievedChunk("a", 0.9, "[Отчёт A]", {}), RetrievedChunk("b", 0.5, "[Отчёт B]", {})]
    text = format_chunks(chunks, threshold=0.8)
    assert "выше порога релевантности (0.8): 1" in text
    assert "[Отчёт A] (score=0.900, релевантный)" in text
    assert "[Отчёт B] (score=0.500, слабо релевантный)" in text


@pytest.mark.skipif(not any(REPORTS.glob("*.pdf")), reason="нет PDF в data/reports (oilgas download-reports)")
def test_chunk_report_keeps_metadata(tmp_path):
    pdf = sorted(REPORTS.glob("*.pdf"))[0]
    s = Settings(data_dir=REPORTS.parent, chunk_size=800, chunk_overlap=100)
    docs = chunk_report(pdf, s)
    assert docs, "нет чанков"
    md = docs[0].metadata
    for key in ("source_file", "publisher", "report_title", "report_date", "page", "label"):
        assert key in md
    assert md["label"].startswith("[Отчёт ") and "стр." in md["label"]
    assert all(len(d.page_content) <= 800 for d in docs)
    assert len({d.metadata["page"] for d in docs}) > 1

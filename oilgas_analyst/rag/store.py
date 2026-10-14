"""Векторное хранилище (Chroma) + индексация PDF-отчётов + поиск top-k."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from oilgas_analyst.config import Settings, get_settings
from oilgas_analyst.rag.catalog import load_catalog, meta_for
from oilgas_analyst.rag.embeddings import E5Embeddings

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float  # косинусная близость, 0..1
    label: str
    metadata: dict


def get_store(settings: Settings | None = None) -> Chroma:
    s = settings or get_settings()
    return _store(str(s.chroma_dir), s.chroma_collection, s.embedding_model, s.embedding_device)


@lru_cache
def _store(chroma_dir: str, collection: str, model: str, device: str) -> Chroma:
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=E5Embeddings(model, device),
        persist_directory=chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception as e:  # битые страницы не должны валить всю индексацию
            log.warning("Не удалось извлечь текст: %s стр. %d: %s", path.name, i, e)
            continue
        if len(text) > 50:
            pages.append((i, text))
    return pages


def chunk_report(path: Path, settings: Settings | None = None) -> list[Document]:
    """Разбивает PDF на чанки с метаданными: отчёт, издатель, дата, страница."""
    s = settings or get_settings()
    meta = meta_for(path.name, load_catalog(s.reports_dir))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for page_no, text in load_pdf_pages(path):
        for idx, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_file": path.name,
                        "publisher": meta.publisher,
                        "report_title": meta.title,
                        "report_short": meta.short_name,
                        "report_date": meta.date,
                        "page": page_no,
                        "chunk_index": idx,
                        "label": meta.label(page_no),
                        "url": meta.url,
                    },
                )
            )
    return docs


def _chunk_id(doc: Document) -> str:
    m = doc.metadata
    key = f"{m['source_file']}|{m['page']}|{m['chunk_index']}|{doc.page_content[:64]}"
    return hashlib.sha1(key.encode()).hexdigest()


def indexed_files(store: Chroma) -> set[str]:
    data = store.get(include=["metadatas"])
    return {m.get("source_file") for m in data.get("metadatas") or [] if m}


def ingest_reports(reports_dir: Path | None = None, force: bool = False, settings: Settings | None = None) -> dict:
    """Индексирует все PDF из каталога отчётов. Уже проиндексированные файлы пропускаются."""
    s = settings or get_settings()
    reports_dir = reports_dir or s.reports_dir
    store = get_store(s)
    done = set() if force else indexed_files(store)
    stats = {"indexed": [], "skipped": [], "chunks": 0}
    for pdf in sorted(reports_dir.glob("*.pdf")):
        if pdf.name in done:
            stats["skipped"].append(pdf.name)
            continue
        if force:
            store.delete(where={"source_file": pdf.name})
        docs = chunk_report(pdf, s)
        if not docs:
            log.warning("В %s не найден текст (скан?) — пропускаю", pdf.name)
            continue
        for i in range(0, len(docs), 256):
            batch = docs[i : i + 256]
            store.add_documents(batch, ids=[_chunk_id(d) for d in batch])
        stats["indexed"].append(pdf.name)
        stats["chunks"] += len(docs)
        log.info("Проиндексирован %s: %d чанков", pdf.name, len(docs))
    return stats


def search_reports(query: str, k: int | None = None, publisher: str | None = None,
                   settings: Settings | None = None) -> list[RetrievedChunk]:
    s = settings or get_settings()
    store = get_store(s)
    k = k or s.rag_top_k
    flt = {"publisher": publisher} if publisher else None
    # Берём с запасом: ежемесячные выпуски одного отчёта содержат почти одинаковые страницы
    results = store.similarity_search_with_score(query, k=k * 4, filter=flt)
    best: dict[str, RetrievedChunk] = {}
    for doc, distance in results:
        # Chroma с cosine-пространством возвращает расстояние 1 - cos
        chunk = RetrievedChunk(doc.page_content, round(1.0 - float(distance), 4),
                               doc.metadata.get("label", ""), doc.metadata)
        key = _signature(chunk)
        prev = best.get(key)
        # Из повторов одного фрагмента в разных выпусках оставляем самый свежий выпуск
        if prev is None or chunk.metadata.get("report_date", "") > prev.metadata.get("report_date", ""):
            if prev is not None:
                chunk.score = max(chunk.score, prev.score)
            best[key] = chunk
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]


def _signature(chunk: RetrievedChunk) -> str:
    """Отпечаток фрагмента без цифр: одна и та же таблица/абзац в соседних выпусках совпадёт."""
    text = re.sub(r"[\d.,%$]+", "", chunk.text.lower())
    text = re.sub(r"\s+", " ", text)
    return f"{chunk.metadata.get('publisher')}|{text[:160]}"


def format_chunks(chunks: list[RetrievedChunk], threshold: float) -> str:
    if not chunks:
        return "База отчётов пуста или ничего не найдено."
    relevant = [c for c in chunks if c.score >= threshold]
    lines = [
        f"Найдено фрагментов: {len(chunks)}, из них выше порога релевантности ({threshold}): {len(relevant)}.",
        "",
    ]
    for i, c in enumerate(chunks, 1):
        tag = "релевантный" if c.score >= threshold else "слабо релевантный"
        lines.append(f"--- Фрагмент {i} {c.label} (score={c.score:.3f}, {tag})")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)

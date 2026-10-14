"""Embeddings на базе multilingual-e5 (sentence-transformers).

Модели семейства E5 обучены с префиксами "query: " / "passage: " —
без них качество поиска заметно падает, поэтому оборачиваем вручную.
Мультиязычная модель нужна потому, что отчёты в основном на английском,
а вопросы пользователей — на русском.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


@lru_cache
def _load_model(name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    @property
    def _model(self):
        return _load_model(self.model_name, self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()

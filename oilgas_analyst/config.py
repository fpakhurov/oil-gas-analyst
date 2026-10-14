"""Конфигурация: все ключи и параметры читаются из окружения / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    # Запуск из репозитория (editable/docker) — data/ рядом с пакетом; иначе — ./data
    repo_data = PROJECT_ROOT / "data"
    return repo_data if (PROJECT_ROOT / "pyproject.toml").exists() else Path.cwd() / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = Field("medium", description="low | medium | high | xhigh | max")
    llm_max_tokens: int = 16000
    # Server-side fallback на другую модель при отказе классификатора безопасности
    llm_refusal_fallback: bool = True
    # Автоматический prompt caching: системный промпт, инструменты и история диалога
    # переиспользуются между шагами агента и ходами диалога (дешевле и быстрее)
    llm_prompt_caching: bool = True
    # Жёсткий дневной лимит расходов на LLM, $ (0 — без лимита)
    daily_budget_usd: float = 10.0

    # --- RAG ---
    # Корень данных; reports/chroma/prices по умолчанию лежат внутри него.
    # При установке пакетом (не из репозитория) задайте DATA_DIR явно.
    data_dir: Path = Field(default_factory=_default_data_dir)
    reports_dir: Path | None = None
    chroma_dir: Path | None = None
    chroma_collection: str = "oilgas_reports"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    rag_top_k: int = 6
    # Порог косинусной близости (e5), выше которого фрагмент считаем «релевантным»
    rag_relevance_threshold: float = 0.84

    # --- Web search ---
    tavily_api_key: str | None = None
    web_max_results: int = 6

    # --- Forecasting ---
    prices_dir: Path | None = None
    price_history_years: int = 12
    # Мировое предложение жидких УВ, млн б/с — база для сценарного модуля
    global_supply_mbd: float = 104.0

    # --- Хранилище приложения ---
    # SQLite: история диалогов (LangGraph checkpointer) и учёт расходов/метаданные
    checkpoints_db: Path | None = None
    app_db: Path | None = None
    # Автообновление базы отчётов и снимка цен, дней (0 — выключено)
    refresh_interval_days: int = 7

    # --- Трейсинг (Langfuse, self-hosted) ---
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    # Адрес для SDK (в docker-compose — http://langfuse-web:3000) и для ссылок в UI
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_ui_url: str = "http://localhost:3000"
    langfuse_project_id: str = "oilgas-analyst"

    @model_validator(mode="after")
    def _derive_paths(self):
        # Плейсхолдеры из .env.example (sk-ant-..., tvly-...) — это «ключ не задан»
        for key in ("anthropic_api_key", "tavily_api_key", "langfuse_public_key", "langfuse_secret_key"):
            value = getattr(self, key)
            if not value or value.endswith("...") or value.strip() in ("", "changeme"):
                setattr(self, key, None)
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.reports_dir = Path(self.reports_dir or self.data_dir / "reports")
        self.chroma_dir = Path(self.chroma_dir or self.data_dir / "chroma")
        self.prices_dir = Path(self.prices_dir or self.data_dir / "prices")
        self.checkpoints_db = Path(self.checkpoints_db or self.data_dir / "checkpoints.sqlite")
        self.app_db = Path(self.app_db or self.data_dir / "app.sqlite")
        return self

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data

WORKDIR /app

# CPU-сборка torch — образ в разы меньше, GPU для e5-small/base не нужен
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY oilgas_analyst ./oilgas_analyst
RUN pip install ".[ui]"

COPY app ./app
COPY .streamlit ./.streamlit
COPY docker/entrypoint.sh /entrypoint.sh
# Отчёты и снимок цен из репозитория (если есть) — базовый корпус для офлайн-старта
COPY data/reports ./data/reports
COPY data/prices ./data/prices
RUN chmod +x /entrypoint.sh

# Модель эмбеддингов скачивается на этапе сборки, чтобы первый запуск не ждал загрузку
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]

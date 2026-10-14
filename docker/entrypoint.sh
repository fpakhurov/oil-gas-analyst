#!/usr/bin/env sh
# Первый запуск: докачиваем отчёты (если папка пуста) и строим индекс. Повторные — мгновенно.
set -e

if ! ls "${DATA_DIR}/reports"/*.pdf >/dev/null 2>&1; then
  echo "[entrypoint] data/reports пуст — скачиваю отчёты…"
  oilgas download-reports --months 3 || echo "[entrypoint] загрузка отчётов не удалась, продолжаю"
fi

echo "[entrypoint] индексация отчётов (новые файлы)…"
oilgas ingest || echo "[entrypoint] индексация не удалась — RAG будет недоступен"

exec "$@"

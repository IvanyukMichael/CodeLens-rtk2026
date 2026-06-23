#!/bin/sh
# CodeLens Docker entrypoint: построить базу (если её ещё нет) и поднять UI.
#
# Режим индексации задаётся переменной INDEX_SCOPE и применяется ТОЛЬКО при
# первом старте (когда базы ещё нет — идемпотентность сохраняется):
#
#   INDEX_SCOPE=full     (по умолчанию) -> python index.py
#       Вся база: gymhero + corpus_extra + corpus_polyglot (168 файлов кода).
#       Демонстрирует требование ТЗ «база ≥80 файлов». Благодаря запечённому в
#       образ кэшу эмбеддингов (cache/index_embeddings.pkl, 100% покрытие) это
#       на CPU = энричмент + запись в Chroma (секунды); bge-m3 для эмбеддинга
#       не грузится — модель скачивается один раз только под финальный smoke-тест.
#
#   INDEX_SCOPE=gymhero               -> python index.py gymhero
#       Только официальная коллекция codelens_gymhero (быстрый/официальный режим).
#
# В ОБОИХ режимах официальный P@5=0.800 считается строго по codelens_gymhero.
set -e

INDEX_SCOPE="${INDEX_SCOPE:-full}"

if [ ! -f chroma_db/chroma.sqlite3 ]; then
  echo "[entrypoint] Первый запуск скачает bge-m3 (~2.3 ГБ) в кэш-том — это может занять время."
  if [ "$INDEX_SCOPE" = "gymhero" ]; then
    echo "[entrypoint] INDEX_SCOPE=gymhero — индексирую ТОЛЬКО gymhero (быстрый/официальный режим) ..."
    python index.py gymhero
  else
    echo "[entrypoint] INDEX_SCOPE=$INDEX_SCOPE — индексирую ВСЮ базу (gymhero + extra, 168 файлов) ..."
    python index.py
  fi
else
  echo "[entrypoint] База ChromaDB найдена — пропускаю индексацию (INDEX_SCOPE=$INDEX_SCOPE не применяется)."
  echo "[entrypoint] Чтобы переиндексировать в другом режиме — очистите том: docker compose down -v"
fi

echo "[entrypoint] Запускаю Streamlit на 0.0.0.0:8501 ..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0

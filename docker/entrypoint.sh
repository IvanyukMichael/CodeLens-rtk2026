#!/bin/sh
# CodeLens Docker entrypoint: построить базу (если её ещё нет) и поднять UI.
set -e

if [ ! -f chroma_db/chroma.sqlite3 ]; then
  echo "[entrypoint] База ChromaDB не найдена — индексирую (python index.py: gymhero + corpus_extra) ..."
  echo "[entrypoint] Первый запуск скачает bge-m3 (~2.3 ГБ) в кэш-том — это может занять время."
  python index.py
else
  echo "[entrypoint] База ChromaDB найдена — пропускаю индексацию."
fi

echo "[entrypoint] Запускаю Streamlit на 0.0.0.0:8501 ..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0

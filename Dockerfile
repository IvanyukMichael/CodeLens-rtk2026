# CodeLens — образ продукта (CPU). Бонус §8 CLAUDE.md: `docker compose up`.
# НЕ заменяет запуск двумя командами — это альтернативный путь развёртывания.
FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Сначала зависимости (кэшируемый слой), потом код.
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY . .

EXPOSE 8501

# entrypoint строит базу (если её нет) и поднимает Streamlit.
ENTRYPOINT ["/bin/sh", "docker/entrypoint.sh"]

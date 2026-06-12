"""Enrichment чанков (§4.1 CLAUDE.md).

Вместо голого кода эмбеддим обогащённый текст вида:

    [путь/к/файлу.py] [class РодительскийКласс]   <- класс только если это метод
    def имя(аргументы) -> тип:                     <- сигнатура из AST
    docstring                                      <- если есть
    --- код ---
    <полный исходный код чанка>

Идея: «прибить» к вектору контекст (где это и что это) и docstring на
естественном языке — это закрывает разрыв «вопрос на NL ↔ имена в коде».

Запуск:
    python enriched_chunks.py            # -> cache/enriched_texts.json + пример
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

from extract_chunks import extract_repo

REPO = Path("gymhero")
OUT = Path("cache/enriched_texts.json")


def build_enriched(chunk: dict) -> str:
    """Собирает обогащённый текст одного чанка."""
    head = f"[{chunk['path']}]"
    if chunk["kind"] == "function" and "." in chunk["name"]:
        parent = chunk["name"].rsplit(".", 1)[0]          # ClassName (метод)
        head += f" [class {parent}]"

    parts = [head, chunk["signature"]]
    if chunk["docstring"]:
        parts.append(chunk["docstring"])
    parts.append("--- код ---")
    parts.append(chunk["code"])
    return "\n".join(parts)


def build_enriched_map(chunks: list[dict]) -> dict[str, str]:
    """{chunk_id: enriched_text} для всех чанков (порядок исходный)."""
    return {c["chunk_id"]: build_enriched(c) for c in chunks}


def main() -> None:
    chunks = extract_repo(REPO)
    enriched = build_enriched_map(chunks)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Чанков: {len(enriched)}")
    print(f"Записано: {OUT}")

    print("\n--- пример (метод) ---")
    sample = next(
        (c for c in chunks if c["kind"] == "function" and "." in c["name"]),
        chunks[0],
    )
    print(enriched[sample["chunk_id"]])


if __name__ == "__main__":
    main()

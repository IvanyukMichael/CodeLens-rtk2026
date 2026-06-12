"""CLI-обёртка над enrichment (§4.1 CLAUDE.md): дамп cache/enriched_texts.json.

Сама логика enrichment живёт в общем модуле enrich.py (единый источник правды
для index.py и экспериментов). Здесь — только re-export и демо-скрипт, чтобы
исторические импорты `from enriched_chunks import build_enriched_map`
(eval_runner, ablation_full, hyde) продолжали работать без изменений.

Запуск:
    python enriched_chunks.py            # -> cache/enriched_texts.json + пример
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

from extract_chunks import extract_repo
from enrich import build_enriched, build_enriched_map  # noqa: F401  (re-export)

REPO = Path("gymhero")
OUT = Path("cache/enriched_texts.json")


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

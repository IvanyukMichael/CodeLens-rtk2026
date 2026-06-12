"""Единый модуль enrichment чанков (§4.1 CLAUDE.md).

Канонический источник правды для обогащённого текста чанка. Им пользуются
и эксперименты (eval_runner, ablation, hyde — через re-export в
enriched_chunks.py), и продакшен-индексатор index.py, чтобы то, что мы
эмбеддим в базу, было БАЙТ-в-БАЙТ тем же, на чём заморожена метрика
(bge-m3 + enriched + HyDE-mix, P@5=0.822). Не дублировать эту логику.

Вместо голого кода эмбеддим обогащённый текст вида:

    [путь/к/файлу.py] [class РодительскийКласс]   <- класс только если это метод
    def имя(аргументы) -> тип:                     <- сигнатура из AST
    docstring                                      <- если есть
    --- код ---
    <полный исходный код чанка>

Идея: «прибить» к вектору контекст (где это и что это) и docstring на
естественном языке — это закрывает разрыв «вопрос на NL ↔ имена в коде».
"""


def build_enriched(chunk: dict) -> str:
    """Собирает обогащённый текст одного чанка.

    chunk — словарь из extract_chunks.extract_chunks (поля path, kind, name,
    signature, docstring, code)."""
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

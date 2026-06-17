"""Мультиязычный AST-экстрактор чанков через tree-sitter (бонус §8 CLAUDE.md:
«второй язык JS/Java»).

Даёт чанки той же схемы, что и питоновский [extract_chunks.py], чтобы их можно
было прогнать через общий enrich.py и положить в ту же коллекцию ChromaDB:

    {chunk_id, path, name, kind, signature, start, end, code, docstring, lang}

chunk_id = {relative_path}:{name}:{start_line}; для метода — name = Class.method
(как в питоновском экстракторе, чтобы формат был единым). start_line — 1-based,
строка объявления (как node.lineno у ast).

Поддержаны: JavaScript (.js/.jsx/.mjs) и Java (.java). Грамматики — отдельные
wheel-пакеты tree-sitter-javascript / tree-sitter-java (прекомпилированы).
Если tree-sitter не установлен — extract_repo_ts() возвращает [] (graceful:
питоновский путь продукта от этого не зависит).

Запуск:
    python extract_chunks_ts.py <папка>
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXT_LANG = {".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
            ".java": "java"}

# Узлы-определения по языкам: тип узла tree-sitter -> kind в нашей схеме.
# container=True означает «может содержать вложенные методы» (рекурсируем с префиксом).
_NODE_KINDS = {
    "javascript": {
        "class_declaration": ("class", True),
        "function_declaration": ("function", False),
        "generator_function_declaration": ("function", False),
        "method_definition": ("function", False),
    },
    "java": {
        "class_declaration": ("class", True),
        "interface_declaration": ("class", True),
        "enum_declaration": ("class", True),
        "method_declaration": ("function", False),
        "constructor_declaration": ("function", False),
    },
}


def _get_parser(lang_name: str):
    """Возвращает Parser для языка или None, если грамматика недоступна."""
    try:
        from tree_sitter import Language, Parser
        if lang_name == "javascript":
            import tree_sitter_javascript as ts
        elif lang_name == "java":
            import tree_sitter_java as ts
        else:
            return None
        return Parser(Language(ts.language()))
    except Exception:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_of(node, src: bytes) -> str | None:
    nm = node.child_by_field_name("name")
    return _text(nm, src) if nm is not None else None


def _leading_doc(node, src: bytes) -> str:
    """JSDoc/Javadoc-комментарий прямо перед объявлением (если есть, вплотную)."""
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "comment":
        if node.start_point[0] - prev.end_point[0] <= 1:      # вплотную (без пустых строк)
            raw = _text(prev, src).strip()
            cleaned = raw.lstrip("/*").rstrip("*/").strip()
            return " ".join(ln.strip(" *") for ln in cleaned.splitlines()).strip()
    return ""


def extract_chunks_ts(path: Path, repo_root: Path) -> list[dict]:
    """Чанки из одного .js/.java файла. [] если язык не поддержан или нет грамматики."""
    lang_name = EXT_LANG.get(path.suffix.lower())
    if lang_name is None:
        return []
    parser = _get_parser(lang_name)
    if parser is None:
        return []

    rel = path.relative_to(repo_root).as_posix()
    src = path.read_bytes()
    lines = src.decode("utf-8", errors="replace").splitlines()
    tree = parser.parse(src)
    kinds = _NODE_KINDS[lang_name]
    out = []

    def emit(node, qualname, kind):
        start = node.start_point[0] + 1                       # 1-based строка объявления
        end = node.end_point[0] + 1
        code = "\n".join(lines[start - 1:end])
        first_line = (lines[start - 1].strip() if start - 1 < len(lines) else "")
        out.append({
            "chunk_id": f"{rel}:{qualname}:{start}",
            "path": rel, "name": qualname, "kind": kind,
            "signature": first_line,                          # строка объявления как сигнатура
            "start": start, "end": end, "code": code,
            "docstring": _leading_doc(node, src),
            "lang": lang_name,
        })

    def walk(node, prefix=""):
        for child in node.children:
            spec = kinds.get(child.type)
            if spec is not None:
                kind, container = spec
                name = _name_of(child, src)
                if name is None:
                    walk(child, prefix)                       # безымянный — спускаемся глубже
                    continue
                qual = prefix + name
                emit(child, qual, kind)
                if container:
                    walk(child, qual + ".")                   # методы получают Class.
            else:
                walk(child, prefix)                           # не-определение: ищем глубже

    walk(tree.root_node)
    return out


def extract_repo_ts(repo_root: Path) -> list[dict]:
    """Все .js/.java под repo_root. [] если tree-sitter не установлен (graceful)."""
    chunks = []
    for path in sorted(repo_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXT_LANG:
            chunks.extend(extract_chunks_ts(path, repo_root))
    return chunks


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus_polyglot")
    if not target.exists():
        print(f"Папка не найдена: {target.resolve()}")
        sys.exit(1)
    files = [p for p in sorted(target.rglob("*"))
             if p.is_file() and p.suffix.lower() in EXT_LANG]
    chunks = extract_repo_ts(target)
    print(f"Папка:        {target.resolve()}")
    print(f"Файлов JS/Java: {len(files)}")
    print(f"Чанков всего: {len(chunks)}")
    for c in chunks[:8]:
        print(f"  [{c['lang']:<10}] {c['chunk_id']}  ({c['kind']})")


if __name__ == "__main__":
    main()

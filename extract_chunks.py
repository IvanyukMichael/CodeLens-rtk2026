"""Корректный AST-экстрактор чанков (эталон из §3 CLAUDE.md).

Чанк = одна функция/класс/метод. chunk_id имеет формат
    {relative_path}:{name}:{start_line}
где relative_path считается от корня репозитория (внешняя папка),
а для метода name квалифицируется как ClassName.method_name.

Запуск:
    python extract_chunks.py <папка>
По умолчанию <папка> = gymhero.
"""

import ast
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _signature(node) -> str:
    """Восстанавливает сигнатуру из AST-узла: 'def name(args) -> ret:' /
    'async def ...' / 'class Name(bases):'. Детерминированно, через ast.unparse."""
    if isinstance(node, ast.ClassDef):
        parts = [ast.unparse(b) for b in node.bases]
        parts += [ast.unparse(k) for k in node.keywords]
        inside = ", ".join(parts)
        return f"class {node.name}({inside}):" if inside else f"class {node.name}:"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){ret}:"


def extract_chunks(py_file: Path, repo_root: Path) -> list[dict]:
    """Возвращает список чанков {chunk_id, path, name, kind, signature, start, end, code, docstring}."""
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []

    def emit(node, qualname, kind):
        start = node.lineno                      # 1-based, строка def/class (как в эталоне)
        end = getattr(node, "end_lineno", start)
        code = "\n".join(lines[start - 1:end])
        out.append({
            "chunk_id": f"{rel}:{qualname}:{start}",
            "path": rel, "name": qualname, "kind": kind,
            "signature": _signature(node),
            "start": start, "end": end, "code": code,
            "docstring": ast.get_docstring(node) or "",
        })

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                emit(child, prefix + child.name, "class")
                walk(child, prefix + child.name + ".")     # методы получают ClassName.
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit(child, prefix + child.name, "function")
                # не спускаемся во вложенные функции — они редко являются эталонами

    walk(tree)
    return out


def extract_repo(repo_root: Path) -> list[dict]:
    """Обходит все .py файлы под repo_root и собирает чанки.

    repo_root — папка, переданная пользователем; это и есть «корень
    репозитория». relative_path в chunk_id считается ОТ НЕЁ САМОЙ, поэтому
    файл на диске gymhero/gymhero/api/dependencies.py даёт путь
    gymhero/api/dependencies.py (внутренний пакет тоже называется gymhero) —
    точно как в эталонных chunk_id."""
    chunks = []
    for py_file in sorted(repo_root.rglob("*.py")):
        chunks.extend(extract_chunks(py_file, repo_root))
    return chunks


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "gymhero")
    if not target.exists():
        print(f"Папка не найдена: {target.resolve()}")
        sys.exit(1)

    py_files = sorted(target.rglob("*.py"))
    chunks = extract_repo(target)

    print(f"Папка:        {target.resolve()}")
    print(f"Файлов .py:   {len(py_files)}")
    print(f"Чанков всего: {len(chunks)}")
    print("\nПервые 5 чанков:")
    for i, c in enumerate(chunks[:5], 1):
        doc = c["docstring"].splitlines()[0] if c["docstring"] else ""
        print(f"\n[{i}] chunk_id = {c['chunk_id']}")
        print(f"    kind={c['kind']}  lines={c['start']}-{c['end']}")
        if doc:
            print(f"    docstring: {doc}")


if __name__ == "__main__":
    main()

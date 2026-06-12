"""CodeLens — индексатор кодовой базы в ChromaDB.

Запуск (одна из двух обязательных команд продукта, см. §8 CLAUDE.md):

    python index.py <папка>          # <папка> = repo_root, по умолчанию gymhero

Пайплайн (строит ТОЛЬКО базу; HyDE и сам поиск — в search-модуле для app.py):

    обход .py  ->  extract_chunks (AST, §3)  ->  enrich (§4.1, общий enrich.py)
        ->  bge-m3 embed (normalize=True)  ->  ChromaDB (persistent, chroma_db/)

Хранилище — ChromaDB (требование ТЗ), space=cosine. ID документа = chunk_id;
в метаданных лежит всё, что нужно UI: path, name, kind, start_line, end_line,
docstring, code, enriched_text. Конфигурация ретрива заморожена на
bge-m3 + enriched (P@5=0.822 с HyDE-mix на стороне запроса) — здесь мы
эмбеддим РОВНО тот же enriched-текст, что в экспериментах, чтобы база была
байт-в-байт совместима с замороженной метрикой.

Идемпотентность: повторный запуск пересоздаёт коллекцию с нуля (без дублей).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
from pathlib import Path

import numpy as np

from extract_chunks import extract_repo
from enrich import build_enriched_map

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "BAAI/bge-m3"
DB_DIR = Path("chroma_db")
COLLECTION = "codelens"
ADD_BATCH = 1000          # размер батча при заливке в Chroma (хватает с запасом)
TEST_QUERY = "как создаётся токен доступа"


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_index(repo_root: Path):
    """Строит коллекцию ChromaDB из всех .py под repo_root. Возвращает
    (collection, model, chunks, n_files, elapsed_sec)."""
    py_files = sorted(repo_root.rglob("*.py"))

    print(f"[1/4] Извлекаю чанки из {repo_root} (AST) ...")
    chunks = extract_repo(repo_root)
    if not chunks:
        print("Чанков не найдено — нечего индексировать.")
        sys.exit(1)
    enriched = build_enriched_map(chunks)            # {chunk_id: enriched_text}
    print(f"      файлов .py: {len(py_files)},  чанков: {len(chunks)}")

    device = _device()
    print(f"[2/4] Загружаю модель {MODEL_NAME} (device={device}) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("[3/4] Эмбеддинги enriched-чанков (bge-m3, normalize=True) ...")
    docs = [enriched[c["chunk_id"]] for c in chunks]
    emb = np.asarray(
        model.encode(docs, normalize_embeddings=True, batch_size=16,
                     show_progress_bar=True),
        dtype=np.float32,
    )

    print(f"[4/4] Пересоздаю коллекцию '{COLLECTION}' в {DB_DIR}/ ...")
    import chromadb
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:                                              # идемпотентность: чистое пересоздание
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "model": MODEL_NAME},
    )

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "path": c["path"],
        "name": c["name"],
        "kind": c["kind"],
        "start_line": c["start"],
        "end_line": c["end"],
        "docstring": c["docstring"],
        "code": c["code"],
        "enriched_text": enriched[c["chunk_id"]],
    } for c in chunks]

    for s in range(0, len(ids), ADD_BATCH):
        e = s + ADD_BATCH
        collection.add(
            ids=ids[s:e],
            embeddings=emb[s:e].tolist(),
            documents=docs[s:e],                      # храним то, что эмбеддили
            metadatas=metadatas[s:e],
        )

    return collection, model, chunks, len(py_files), client


def smoke_query(collection, model, query: str, k: int = 3) -> None:
    """Тестовый ВЕКТОРНЫЙ поиск прямо из ChromaDB (без HyDE) — sanity-check базы."""
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=k)
    ids = res["ids"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]

    print(f"\nТестовый запрос: {query!r}  ->  top-{k}")
    print("-" * 78)
    for rank, (cid, dist, m) in enumerate(zip(ids, dists, metas), 1):
        rel = (1.0 - dist) * 100.0                    # space=cosine: distance = 1 - cos
        doc = (m.get("docstring") or "").splitlines()
        doc = doc[0] if doc else ""
        print(f"{rank}. [{rel:5.1f}%] {cid}")
        print(f"   {m['kind']}  {m['path']}  L{m['start_line']}-{m['end_line']}")
        if doc:
            print(f"   docstring: {doc}")
    print("-" * 78)


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "gymhero")
    if not target.exists():
        print(f"Папка не найдена: {target.resolve()}")
        sys.exit(1)

    t0 = time.perf_counter()
    collection, model, chunks, n_files, _client = build_index(target)
    elapsed = time.perf_counter() - t0

    n = collection.count()
    print("\n" + "=" * 78)
    print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print(f"  папка:            {target.resolve()}")
    print(f"  файлов .py:       {n_files}")
    print(f"  чанков в базе:    {n}")
    print(f"  хранилище:        {DB_DIR.resolve()}  (коллекция '{COLLECTION}', space=cosine)")
    print(f"  модель:           {MODEL_NAME}")
    print(f"  время индексации: {elapsed:.1f} c")
    print("=" * 78)

    smoke_query(collection, model, TEST_QUERY)


if __name__ == "__main__":
    main()

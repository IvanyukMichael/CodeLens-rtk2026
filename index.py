"""CodeLens — индексатор кодовой базы в ChromaDB.

Запуск (одна из двух обязательных команд продукта, см. §8 CLAUDE.md):

    python index.py                  # вся база по умолчанию: gymhero + corpus_extra
    python index.py <папка> [...]    # индексировать только указанные корни

Пайплайн (строит ТОЛЬКО базу; HyDE и сам поиск — в search-модуле для app.py):

    обход .py  ->  extract_chunks (AST, §3)  ->  enrich (§4.1, общий enrich.py)
        ->  bge-m3 embed (normalize=True)  ->  ChromaDB (persistent, chroma_db/)

Несколько корней (требование ТЗ «база ≥80 файлов»). relative_path в chunk_id
считается ОТ КАЖДОГО корня по отдельности, поэтому gymhero сохраняет точные
chunk_id вида `gymhero/...` (калибровка 30/30), а доп. корпус получает
собственные префиксы (`click/...`, `rich/...`) — пересечений нет. Официальная
метрика считается только по gymhero; доп. репозитории — это размер базы,
масштаб и «отвлекающие» чанки для честной нагрузки на ретрив.

Хранилище — ChromaDB (требование ТЗ), space=cosine. ID документа = chunk_id;
в метаданных лежит всё, что нужно UI: path, name, kind, start_line, end_line,
docstring, code, enriched_text, source (репозиторий-источник). Конфигурация
ретрива: bge-m3 + enriched (P@5=0.800 офиц.15 / 0.831 расш.71 с HyDE-mix temp=0
на стороне запроса) — здесь мы эмбеддим РОВНО тот же enriched-текст, что в
экспериментах, чтобы база была байт-в-байт совместима с метрикой.

Идемпотентность: повторный запуск пересоздаёт коллекцию с нуля (без дублей).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов HF

import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np

from extract_chunks import extract_repo
from extract_chunks_ts import extract_repo_ts        # JS/Java через tree-sitter (graceful)
from enrich import build_enriched_map

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "BAAI/bge-m3"
DB_DIR = Path("chroma_db")
COLLECTION = "codelens"
ADD_BATCH = 1000          # размер батча при заливке в Chroma (хватает с запасом)
TEST_QUERY = "как создаётся токен доступа"

# Кэш эмбеддингов чанков: {sha1(enriched_text): vector}. Ключ — по СОДЕРЖИМОМУ
# enriched-текста, поэтому при неизменном коде повторная индексация переиспользует
# вектора (на CPU это превращает минуты в секунды), а изменённые/новые чанки
# пересчитываются автоматически. Если ВСЁ из кэша — модель bge-m3 даже не грузится.
EMB_CACHE = Path("cache/index_embeddings.pkl")

# Вся база по умолчанию (≥80 файлов): gymhero (официальный корпус, точные
# chunk_id) + corpus_extra (click/rich, дальние по домену «отвлекающие» чанки)
# + corpus_polyglot (демо второго языка: JS/Java через tree-sitter).
DEFAULT_ROOTS = ["gymhero", "corpus_extra", "corpus_polyglot"]

# Расширения, считающиеся индексируемыми файлами (для счётчика «≥80 файлов»).
CODE_EXTS = (".py", ".js", ".jsx", ".mjs", ".java")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _source_of(path: str) -> str:
    """Репозиторий-источник чанка = первый компонент относительного пути
    (gymhero / click / rich). Нужен UI для фильтра и scoped-метрике."""
    return path.split("/", 1)[0]


def _emb_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_emb_cache(model_name: str) -> dict:
    """{sha1(text): list[float]} для данной модели. {} при промахе/несовместимости."""
    if EMB_CACHE.exists():
        try:
            data = pickle.loads(EMB_CACHE.read_bytes())
            if data.get("model") == model_name:
                return data.get("emb", {})
        except Exception:
            pass
    return {}


def _save_emb_cache(model_name: str, emb_by_hash: dict) -> None:
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.write_bytes(pickle.dumps({"model": model_name, "emb": emb_by_hash}))


def embed_docs(docs: list[str]):
    """Эмбеддинги enriched-текстов с кэшем по содержимому. Возвращает (emb, model).
    model=None, если всё взято из кэша (bge-m3 не грузился)."""
    hashes = [_emb_hash(d) for d in docs]
    cache = _load_emb_cache(MODEL_NAME)
    miss = [i for i, h in enumerate(hashes) if h not in cache]
    print(f"      кэш эмбеддингов: {len(docs) - len(miss)}/{len(docs)} попаданий, "
          f"посчитать: {len(miss)}")

    model = None
    if miss:
        device = _device()
        print(f"[2/4] Загружаю модель {MODEL_NAME} (device={device}) ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"[3/4] Эмбеддинги {len(miss)} новых чанков (bge-m3, normalize=True) ...")
        new = np.asarray(
            model.encode([docs[i] for i in miss], normalize_embeddings=True,
                         batch_size=16, show_progress_bar=True),
            dtype=np.float32)
        for j, i in enumerate(miss):
            cache[hashes[i]] = new[j].tolist()
        _save_emb_cache(MODEL_NAME, cache)
    else:
        print("[2/4] Все эмбеддинги из кэша — модель не загружается.")
        print("[3/4] Сборка матрицы эмбеддингов из кэша ...")

    emb = np.asarray([cache[h] for h in hashes], dtype=np.float32)
    return emb, model


def collect_chunks(roots: list[Path]) -> tuple[list[dict], int]:
    """Извлекает чанки из нескольких корней. relative_path считается ОТ КАЖДОГО
    корня (см. extract_repo), поэтому gymhero сохраняет `gymhero/...`, а доп.
    корпус — свои префиксы. Python — через AST (extract_repo), JS/Java — через
    tree-sitter (extract_repo_ts, graceful: [] если грамматик нет). Дубли
    chunk_id (на всякий случай) отбрасываются. Возвращает (chunks, n_files)."""
    chunks, seen, n_files = [], set(), 0
    for root in roots:
        n_files += sum(1 for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in CODE_EXTS)
        for c in list(extract_repo(root)) + list(extract_repo_ts(root)):
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            chunks.append(c)
    return chunks, n_files


def build_index(roots: list[Path]):
    """Строит коллекцию ChromaDB из всех .py под перечисленными корнями.
    Возвращает (collection, model, chunks, n_files, client)."""
    print(f"[1/4] Извлекаю чанки из {', '.join(str(r) for r in roots)} (AST) ...")
    chunks, n_files = collect_chunks(roots)
    if not chunks:
        print("Чанков не найдено — нечего индексировать.")
        sys.exit(1)
    enriched = build_enriched_map(chunks)            # {chunk_id: enriched_text}
    print(f"      файлов .py: {n_files},  чанков: {len(chunks)}")

    docs = [enriched[c["chunk_id"]] for c in chunks]
    emb, model = embed_docs(docs)                     # кэш по содержимому; model=None если всё из кэша

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
        "source": _source_of(c["path"]),             # gymhero / click / rich (для UI и scoped)
        "lang": c.get("lang", "python"),             # python (AST) / javascript / java (tree-sitter)
    } for c in chunks]

    for s in range(0, len(ids), ADD_BATCH):
        e = s + ADD_BATCH
        collection.add(
            ids=ids[s:e],
            embeddings=emb[s:e].tolist(),
            documents=docs[s:e],                      # храним то, что эмбеддили
            metadatas=metadatas[s:e],
        )

    return collection, model, chunks, n_files, client


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
    args = sys.argv[1:]
    roots = [Path(a) for a in args] if args else [Path(r) for r in DEFAULT_ROOTS]
    roots = [r for r in roots if r.exists()]
    if not roots:
        missing = args or DEFAULT_ROOTS
        print(f"Папки не найдены: {', '.join(map(str, missing))}")
        sys.exit(1)

    t0 = time.perf_counter()
    collection, model, chunks, n_files, _client = build_index(roots)
    elapsed = time.perf_counter() - t0

    n = collection.count()
    print("\n" + "=" * 78)
    print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print(f"  корни:            {', '.join(str(r.resolve()) for r in roots)}")
    print(f"  файлов .py:       {n_files}")
    print(f"  чанков в базе:    {n}")
    print(f"  хранилище:        {DB_DIR.resolve()}  (коллекция '{COLLECTION}', space=cosine)")
    print(f"  модель:           {MODEL_NAME}")
    print(f"  время индексации: {elapsed:.1f} c")
    print("=" * 78)

    if model is None:                                 # всё пришло из кэша — грузим bge-m3 для smoke-теста
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=_device())
    smoke_query(collection, model, TEST_QUERY)


if __name__ == "__main__":
    main()

"""CodeLens — индексатор кодовой базы в ChromaDB.

Запуск (одна из двух обязательных команд продукта, см. §8 CLAUDE.md):

    python index.py                  # строит ОБЕ коллекции: gymhero + доп.корпус
    python index.py <папка> [...]    # индексировать только указанные корни

Пайплайн (строит ТОЛЬКО базу; HyDE и сам поиск — в search-модуле для app.py):

    обход .py  ->  extract_chunks (AST, §3)  ->  enrich (§4.1, общий enrich.py)
        ->  bge-m3 embed (normalize=True)  ->  ChromaDB (persistent, chroma_db/)

ДВЕ ИЗОЛИРОВАННЫЕ КОЛЛЕКЦИИ (ключевое архитектурное решение, см. REPORT
«Режимы индексации»): gymhero идёт в codelens_gymhero, доп. репозитории — в
codelens_extra. Официальная метрика P@5 считается СТРОГО по codelens_gymhero,
поэтому она не зависит от размера общей базы и не подвержена cross-project
dilution (доменно-близкие чанки из другого проекта не вытесняют эталонные
gymhero-ответы из топ-5; см. REPORT, замер A). Требование ТЗ «база ≥80 файлов»
закрывается суммой обеих коллекций (168 файлов).

relative_path в chunk_id считается ОТ КАЖДОГО корня по отдельности, поэтому
gymhero сохраняет точные chunk_id вида `gymhero/...` (калибровка 30/30), а доп.
корпус получает собственные префиксы (`click/...`, `rich/...`) — пересечений нет.

Хранилище — ChromaDB (требование ТЗ), space=cosine. ID документа = chunk_id;
в метаданных лежит всё, что нужно UI: path, name, kind, start_line, end_line,
docstring, code, enriched_text, source (репозиторий-источник). Конфигурация
ретрива: bge-m3 + enriched (P@5=0.800 офиц.15 / 0.831 расш.71 с HyDE-mix temp=0
на стороне запроса) — здесь мы эмбеддим РОВНО тот же enriched-текст, что в
экспериментах, чтобы база была байт-в-байт совместима с метрикой.

Идемпотентность: повторный запуск пересоздаёт gymhero-коллекцию с нуля (без
дублей). Для codelens_extra пересоздаётся только ВСТРОЕННЫЙ корпус — добавленные
через вкладку «Репозитории» пользовательские репозитории (origin="user")
переживают `python index.py` и управляются отдельно (add_repo.py). См.
_reset_for_builtin.
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

# ДВЕ ИЗОЛИРОВАННЫЕ КОЛЛЕКЦИИ (см. REPORT «Режимы индексации»):
#   gymhero  -> codelens_gymhero  — официальный корпус, точные chunk_id, эталон
#                                   метрики. По НЕЙ считается P@5 (офиц.15 / расш.).
#   доп.репо -> codelens_extra    — click/rich + JS/Java-демо. Это размер базы
#                                   (≥80 файлов ТЗ) и мультипроектный поиск в UI.
# gymhero и distractor НИКОГДА не лежат в одной коллекции: официальная метрика
# 0.800 считается строго по codelens_gymhero и НЕ ЗАВИСИТ от размера общей базы.
# Это устраняет cross-project dilution на эталонных вопросах (см. REPORT, замер A).
GYMHERO_COLLECTION = "codelens_gymhero"
EXTRA_COLLECTION = "codelens_extra"

ADD_BATCH = 1000          # размер батча при заливке в Chroma (хватает с запасом)
TEST_QUERY = "как создаётся токен доступа"

# Кэш эмбеддингов чанков: {sha1(enriched_text): vector}. Ключ — по СОДЕРЖИМОМУ
# enriched-текста, поэтому при неизменном коде повторная индексация переиспользует
# вектора (на CPU это превращает минуты в секунды), а изменённые/новые чанки
# пересчитываются автоматически. Если ВСЁ из кэша — модель bge-m3 даже не грузится.
EMB_CACHE = Path("cache/index_embeddings.pkl")

# Маршрутизация корней по коллекциям (см. выше). gymhero -> своя коллекция,
# всё остальное -> extra. Так `python index.py` без аргументов строит ОБЕ
# коллекции раздельно, а официальная gymhero-коллекция физически не может быть
# затёрта distractor'ами (раньше DEFAULT_ROOTS сваливал всё в одну `codelens`).
GYMHERO_ROOTS = ["gymhero"]
EXTRA_ROOTS = ["corpus_extra", "corpus_polyglot"]

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


def _reset_for_builtin(client, collection_name: str):
    """Готовит коллекцию к перезаписи ВСТРОЕННЫМ корпусом и возвращает её.

    Граница владения (вариант B изоляции): коллекция codelens_extra может
    содержать пользовательские репозитории (origin="user"), добавленные через
    вкладку «Репозитории»/add_repo.py. Поэтому для extra мы НЕ удаляем коллекцию
    целиком, а чистим только встроенные чанки (origin != "user") по их id —
    пользовательские репозитории ПЕРЕЖИВАЮТ `python index.py`. Старые чанки без
    поля origin трактуются как встроенные (удаляются), что корректно мигрирует
    базу, построенную прошлой версией индексатора.

    Для gymhero (и любой другой коллекции) — прежнее чистое пересоздание: там
    user-чанков нет по инварианту изоляции, а официальная метрика требует
    байт-в-байт детерминированной перестройки."""
    meta = {"hnsw:space": "cosine", "model": MODEL_NAME}
    if collection_name != EXTRA_COLLECTION:
        try:                                          # идемпотентность: чистое пересоздание
            client.delete_collection(collection_name)
        except Exception:
            pass
        return client.create_collection(name=collection_name, metadata=meta)

    col = client.get_or_create_collection(name=collection_name, metadata=meta)
    got = col.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    builtin_ids = [i for i, m in zip(ids, metas) if (m or {}).get("origin") != "user"]
    for s in range(0, len(builtin_ids), ADD_BATCH):
        col.delete(ids=builtin_ids[s:s + ADD_BATCH])
    n_user = len(ids) - len(builtin_ids)
    if n_user:
        print(f"      сохранено пользовательских чанков (origin=user): {n_user}")
    return col


def build_index(roots: list[Path], collection_name: str):
    """Строит ОДНУ изолированную коллекцию ChromaDB из всех файлов под
    перечисленными корнями. Возвращает (collection, model, chunks, n_files, client)."""
    print(f"[1/4] Извлекаю чанки из {', '.join(str(r) for r in roots)} (AST) ...")
    chunks, n_files = collect_chunks(roots)
    if not chunks:
        print("Чанков не найдено — нечего индексировать.")
        sys.exit(1)
    enriched = build_enriched_map(chunks)            # {chunk_id: enriched_text}
    print(f"      файлов .py: {n_files},  чанков: {len(chunks)}")

    docs = [enriched[c["chunk_id"]] for c in chunks]
    emb, model = embed_docs(docs)                     # кэш по содержимому; model=None если всё из кэша

    print(f"[4/4] Пересоздаю коллекцию '{collection_name}' в {DB_DIR}/ ...")
    import chromadb
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = _reset_for_builtin(client, collection_name)

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
        "origin": "builtin",                         # встроенный корпус; user-репо помечены "user"
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


def _is_gymhero_root(p: Path) -> bool:
    """Корень относится к официальной gymhero-коллекции, только если это сама
    папка gymhero. Любая другая папка (включая чужой репозиторий судьи) уходит
    в codelens_extra — так официальная коллекция не может быть затёрта."""
    return p.resolve().name == "gymhero"


def _plan_jobs(args: list[str]) -> list[tuple[str, list[Path]]]:
    """Маршрутизация корней по коллекциям. Без аргументов — обе коллекции из
    стандартных корней; с аргументами — только те коллекции, в которые попали
    переданные папки. Возвращает [(collection_name, [roots]), ...]."""
    if args:
        roots = [Path(a) for a in args if Path(a).exists()]
        gym = [r for r in roots if _is_gymhero_root(r)]
        extra = [r for r in roots if not _is_gymhero_root(r)]
    else:
        gym = [Path(r) for r in GYMHERO_ROOTS if Path(r).exists()]
        extra = [Path(r) for r in EXTRA_ROOTS if Path(r).exists()]
    jobs = []
    if gym:
        jobs.append((GYMHERO_COLLECTION, gym))
    if extra:
        jobs.append((EXTRA_COLLECTION, extra))
    return jobs


def main() -> None:
    args = sys.argv[1:]
    jobs = _plan_jobs(args)
    if not jobs:
        target = args or (GYMHERO_ROOTS + EXTRA_ROOTS)
        print(f"Папки не найдены: {', '.join(map(str, target))}")
        sys.exit(1)

    built = []                                        # (collection, model, n_files, n_chunks)
    t0 = time.perf_counter()
    for coll_name, roots in jobs:
        print(f"\n### Коллекция '{coll_name}'  <-  {', '.join(str(r) for r in roots)}")
        collection, model, _chunks, n_files, _client = build_index(roots, coll_name)
        built.append((coll_name, collection, model, n_files, collection.count()))
    elapsed = time.perf_counter() - t0

    total_files = sum(b[3] for b in built)
    total_chunks = sum(b[4] for b in built)
    print("\n" + "=" * 78)
    print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    for coll_name, _c, _m, n_files, n_chunks in built:
        print(f"  коллекция {coll_name:18} файлов={n_files:4}  чанков={n_chunks:4}")
    print(f"  ИТОГО по базе:    файлов={total_files} (≥80 ТЗ: "
          f"{'OK' if total_files >= 80 else 'НЕ ВЫПОЛНЕНО'}),  чанков={total_chunks}")
    print(f"  хранилище:        {DB_DIR.resolve()}  (space=cosine)")
    print(f"  модель:           {MODEL_NAME}")
    print(f"  время индексации: {elapsed:.1f} c")
    print(f"  ВАЖНО:            официальный P@5 считается ТОЛЬКО по "
          f"{GYMHERO_COLLECTION} (изоляция от distractor).")
    print("=" * 78)

    # smoke-тест на gymhero-коллекции, если она строилась (иначе — на первой)
    sc = next((b for b in built if b[0] == GYMHERO_COLLECTION), built[0])
    coll_name, collection, model = sc[0], sc[1], sc[2]
    if model is None:                                 # всё из кэша — грузим bge-m3 для smoke-теста
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=_device())
    print(f"\nSmoke-тест по коллекции '{coll_name}':")
    smoke_query(collection, model, TEST_QUERY)


if __name__ == "__main__":
    main()

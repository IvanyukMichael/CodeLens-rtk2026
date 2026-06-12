"""CodeLens — поисковое ядро (финальная замороженная конфигурация).

Конфигурация ретрива: bge-m3 + enriched (в базе) + HyDE-mix на стороне запроса
(α=0.5), P@5=0.822. Читает коллекцию `codelens`, построенную index.py.

    from search import search
    res = search("как создаётся токен доступа", top_k=5, use_hyde=True)

Логика запроса:
    q_emb = bge-m3(query)                                  (normalize=True)
    если use_hyde и Ollama доступна:
        hyde = qwen2.5-coder:7b(query)   # тот же промт/функция, что в hyde.py
        h_emb = bge-m3(hyde)
        qvec = 0.5*q_emb + 0.5*h_emb     # mix-логика финальной конфигурации
    иначе:
        qvec = q_emb                     # graceful fallback, демо не падает
    top_k в ChromaDB (space=cosine) по qvec  ->  relevance% = (1 - distance)*100

HyDE-генерации кэшируются по тексту запроса (cache/hyde_query_cache.json) —
повторные запросы мгновенны и не зависят от Ollama.

Промт HyDE и сам вызов LLM переиспользуются из hyde.py (не дублируем).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов HF

import json
import os
import time
from pathlib import Path

import numpy as np

# Переиспользуем промт и вызов Ollama из экспериментального hyde.py (не переписываем).
from hyde import ollama_generate, LLM_MODEL

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "BAAI/bge-m3"
DB_DIR = Path("chroma_db")
COLLECTION = "codelens"
ALPHA = 0.5                                   # вес запроса в mix (1-α — вес HyDE)
HYDE_QUERY_CACHE = Path("cache/hyde_query_cache.json")

# Порог «негативного» запроса: лучший relevance ниже — считаем, что кода нет.
NEGATIVE_THRESHOLD = 40.0

_MODEL = None
_COLLECTION = None
_HYDE_CACHE = None


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model():
    """bge-m3, загружается один раз на процесс."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, device=_device())
    return _MODEL


def get_collection():
    """Коллекция ChromaDB `codelens`, открывается один раз на процесс."""
    global _COLLECTION
    if _COLLECTION is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _COLLECTION = client.get_collection(COLLECTION)
    return _COLLECTION


def warmup():
    """Прогреть модель и коллекцию (для @st.cache_resource в app.py)."""
    return get_model(), get_collection()


def _load_hyde_cache() -> dict:
    global _HYDE_CACHE
    if _HYDE_CACHE is None:
        if HYDE_QUERY_CACHE.exists():
            _HYDE_CACHE = json.loads(HYDE_QUERY_CACHE.read_text(encoding="utf-8"))
        else:
            _HYDE_CACHE = {}
    return _HYDE_CACHE


def _save_hyde_cache() -> None:
    HYDE_QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HYDE_QUERY_CACHE.write_text(
        json.dumps(_HYDE_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_hyde(query: str):
    """Возвращает (hyde_text, from_cache, error). hyde_text=None ⇒ HyDE недоступен.

    Никогда не бросает: если Ollama недоступна — error заполнен, поиск идёт без HyDE.
    """
    cache = _load_hyde_cache()
    if query in cache:
        return cache[query], True, None
    try:
        text = ollama_generate(query)
    except Exception as e:                          # ConnectionError/Timeout/… — fallback
        return None, False, f"{type(e).__name__}: {e}"
    if not text.strip():
        return None, False, "пустой ответ LLM"
    cache[query] = text
    _save_hyde_cache()
    return text, False, None


def search(query: str, top_k: int = 5, use_hyde: bool = True) -> dict:
    """Семантический поиск по коду. Возвращает dict с результатами, latency и
    флагами HyDE.

    results[i] = {chunk_id, path, name, kind, start, end, code, docstring, relevance}
    relevance — косинусная близость в процентах (0..100), отсортировано по убыванию.
    """
    model = get_model()
    collection = get_collection()

    out = {
        "query": query, "top_k": top_k,
        "use_hyde_requested": use_hyde, "hyde_used": False,
        "hyde_from_cache": False, "hyde_text": None, "warning": None,
        "results": [],
        "latency": {"hyde_ms": 0.0, "embed_ms": 0.0, "search_ms": 0.0, "total_ms": 0.0},
    }
    if not query or not query.strip():
        return out

    t_all = time.perf_counter()

    # --- HyDE (опционально, с graceful fallback) ---
    hyde_text = None
    if use_hyde:
        t0 = time.perf_counter()
        hyde_text, from_cache, err = _get_hyde(query)
        out["latency"]["hyde_ms"] = (time.perf_counter() - t0) * 1000
        if hyde_text is not None:
            out["hyde_used"] = True
            out["hyde_from_cache"] = from_cache
            out["hyde_text"] = hyde_text
        else:
            out["warning"] = (f"Ollama ({LLM_MODEL}) недоступна — поиск без HyDE "
                              f"(только эмбеддинг запроса). {err}")

    # --- эмбеддинг запроса (+ HyDE) одной батч-кодировкой ---
    t0 = time.perf_counter()
    texts = [query] + ([hyde_text] if hyde_text is not None else [])
    embs = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    if hyde_text is not None:
        qvec = ALPHA * embs[0] + (1.0 - ALPHA) * embs[1]    # mix-логика финала
    else:
        qvec = embs[0]
    out["latency"]["embed_ms"] = (time.perf_counter() - t0) * 1000

    # --- поиск в ChromaDB ---
    t0 = time.perf_counter()
    res = collection.query(
        query_embeddings=[qvec.tolist()], n_results=top_k,
        include=["metadatas", "distances"])
    out["latency"]["search_ms"] = (time.perf_counter() - t0) * 1000

    ids = res["ids"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    for cid, dist, m in zip(ids, dists, metas):
        rel = max(0.0, min(100.0, (1.0 - dist) * 100.0))   # space=cosine
        out["results"].append({
            "chunk_id": cid,
            "path": m["path"], "name": m["name"], "kind": m["kind"],
            "start": m["start_line"], "end": m["end_line"],
            "code": m["code"], "docstring": m.get("docstring", ""),
            "relevance": rel,
        })

    out["latency"]["total_ms"] = (time.perf_counter() - t_all) * 1000
    return out


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CodeLens search (CLI)")
    ap.add_argument("query", nargs="*", help="текст запроса")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--no-hyde", action="store_true")
    args = ap.parse_args()

    query = " ".join(args.query) or "как создаётся токен доступа"
    r = search(query, top_k=args.top_k, use_hyde=not args.no_hyde)

    if r["warning"]:
        print(f"[!] {r['warning']}")
    lat = r["latency"]
    print(f"Запрос: {query!r}  (HyDE={'on' if r['hyde_used'] else 'off'}"
          f"{', cache' if r['hyde_from_cache'] else ''})")
    print(f"latency: hyde={lat['hyde_ms']:.0f} ms, embed={lat['embed_ms']:.0f} ms, "
          f"search={lat['search_ms']:.1f} ms, total={lat['total_ms']:.0f} ms")
    print("-" * 78)
    for i, hit in enumerate(r["results"], 1):
        print(f"{i}. [{hit['relevance']:5.1f}%] {hit['chunk_id']}")
        print(f"   {hit['kind']}  {hit['path']}  L{hit['start']}-{hit['end']}")


if __name__ == "__main__":
    _cli()

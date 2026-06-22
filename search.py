"""CodeLens — поисковое ядро (финальная конфигурация).

Конфигурация ретрива: bge-m3 + enriched (в базе) + HyDE-mix на стороне запроса
(α=0.5), HyDE greedy при temperature=0 (детерминирован). Воспроизводимо вживую:
P@5 = 0.800 (офиц. 15) / 0.831 (расш. 71), идентично между прогонами. Читает
коллекцию `codelens`, построенную index.py.

    from search import search
    res = search("как создаётся токен доступа", top_k=5, use_hyde=True)

Логика запроса:
    q_emb = bge-m3(query)                                  (normalize=True)
    если use_hyde и Ollama доступна:
        hyde = qwen2.5-coder:7b(query, temp=0)   # тот же промт/функция, что в hyde.py
        h_emb = bge-m3(hyde)
        qvec = 0.5*q_emb + 0.5*h_emb     # mix-логика финальной конфигурации
    иначе:
        qvec = q_emb                     # graceful fallback, демо не падает
    top_k в ChromaDB (space=cosine) по qvec  ->  relevance% = (1 - distance)*100

HyDE-генерации кэшируются по тексту запроса (cache/hyde_query_cache.json) — это
ОБЫЧНЫЙ кэш ускорения: пишется из живых temp=0 генераций, повторные запросы
мгновенны и не зависят от Ollama. Никакого frozen-сида: cache miss => живая
temp=0 генерация (детерминированная), которая ложится в тот же кэш.

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

# Две изолированные коллекции (строит index.py). Источник поиска управляет тем,
# по какой(-им) из них идёт запрос:
#   "gymhero" — только codelens_gymhero. По НЕЙ считается официальная метрика
#               (P@5=0.800), поэтому она не зависит от размера общей базы.
#   "all"     — gymhero + extra (вся база ≥80 файлов): мультипроектный поиск в UI.
#   "extra"   — только codelens_extra (отладка/демо).
# DEFAULT_SOURCE="gymhero" => predict.py и метрики по умолчанию считают на
# официальной коллекции и невозможно случайно подмешать distractor.
GYMHERO_COLLECTION = "codelens_gymhero"
EXTRA_COLLECTION = "codelens_extra"
DEFAULT_SOURCE = "gymhero"
COLLECTION = GYMHERO_COLLECTION               # legacy-алиас для старых импортов

# Какие коллекции опрашивать для каждого источника (порядок не важен — потом
# мёрджим по релевантности).
_SOURCE_COLLECTIONS = {
    "gymhero": [GYMHERO_COLLECTION],
    "extra": [EXTRA_COLLECTION],
    "all": [GYMHERO_COLLECTION, EXTRA_COLLECTION],
}

ALPHA = 0.5                                   # вес запроса в mix (1-α — вес HyDE)
HYDE_QUERY_CACHE = Path("cache/hyde_query_cache.json")

# Порог «негативного» запроса: если лучший relevance ниже — считаем, что код
# отсутствует. Откалиброван НА ДАННЫХ (top-1 relevance по 88 позитивам eval+sample
# против негативов, см. README §3): негативы дают top-1 ≤ 58.4%, истинный «пол»
# позитивов = 62.8% (на 71-q eval), самый слабый позитив (RU-easy) = 65.3%. T=60
# лежит в чистом зазоре 58.4→62.8: ловит все негативы (3/3), ложных «не найдено»
# на позитивах ≤1.1%. Не подгоняем под демо — порог ниже легитимного RU-easy 65.3%.
NEGATIVE_THRESHOLD = 60.0

_MODEL = None
_CLIENT = None
_COLLECTIONS = {}                             # {name: collection}, открываются лениво
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


def _client():
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        _CLIENT = chromadb.PersistentClient(path=str(DB_DIR))
    return _CLIENT


def get_collection(name: str = GYMHERO_COLLECTION):
    """Коллекция ChromaDB по имени, открывается один раз на процесс.
    По умолчанию — официальная gymhero-коллекция."""
    if name not in _COLLECTIONS:
        _COLLECTIONS[name] = _client().get_collection(name)
    return _COLLECTIONS[name]


def available_sources() -> list[str]:
    """Источники, для которых ВСЕ нужные коллекции реально существуют в базе.
    Позволяет UI не предлагать «вся база», если extra-коллекция не построена."""
    existing = {c.name for c in _client().list_collections()}
    return [src for src, names in _SOURCE_COLLECTIONS.items()
            if all(n in existing for n in names)]


def warmup():
    """Прогреть модель и официальную коллекцию (для @st.cache_resource в app.py)."""
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
        text = ollama_generate(query, temperature=0.0)   # greedy => детерминированно
    except Exception as e:                          # ConnectionError/Timeout/… — fallback
        return None, False, f"{type(e).__name__}: {e}"
    if not text.strip():
        return None, False, "пустой ответ LLM"
    cache[query] = text
    _save_hyde_cache()
    return text, False, None


def search(query: str, top_k: int = 5, use_hyde: bool = True,
           source: str = DEFAULT_SOURCE, where: dict | None = None) -> dict:
    """Семантический поиск по коду. Возвращает dict с результатами, latency и
    флагами HyDE.

    source — область поиска: "gymhero" (официальная коллекция, метрика 0.800),
    "all" (вся база ≥80 файлов: gymhero+extra), "extra". Официальный скоринг
    (predict.py, страница метрик) использует "gymhero" => P@5 не зависит от
    размера общей базы. Для "all" результаты двух коллекций мёрджатся по
    релевантности (top-k из каждой достаточно для точного глобального top-k).

    where — опц. metadata-фильтр ChromaDB, пробрасывается в collection.query без
    изменений (напр. {"source": "owner__repo"} для поиска по конкретному
    пользовательскому репозиторию во вкладке «Репозитории»). По умолчанию None —
    поведение байт-в-байт прежнее, официальный путь source="gymhero" (predict.py,
    вкладка «Метрики») его не использует и не затронут.

    results[i] = {chunk_id, path, name, kind, start, end, code, docstring,
    relevance, source}. relevance — косинусная близость в процентах (0..100),
    отсортировано по убыванию.
    """
    model = get_model()
    if source not in _SOURCE_COLLECTIONS:
        source = DEFAULT_SOURCE
    target_names = list(_SOURCE_COLLECTIONS[source])

    out = {
        "query": query, "top_k": top_k, "source": source,
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

    # --- поиск в ChromaDB (одна или несколько коллекций) ---
    t0 = time.perf_counter()
    qlist = qvec.tolist()
    merged = []                                   # (dist, cid, meta) по всем коллекциям
    missing = []
    for name in target_names:
        try:
            col = get_collection(name)
        except Exception:
            missing.append(name)                  # коллекция не построена — пропускаем
            continue
        res = col.query(query_embeddings=[qlist], n_results=top_k, where=where,
                        include=["metadatas", "distances"])
        merged.extend(zip(res["distances"][0], res["ids"][0], res["metadatas"][0]))
    # глобальный top-k по косинусной дистанции (меньше = релевантнее). Взять top-k
    # из каждой коллекции достаточно: вне топ-k своей коллекции чанк не может
    # попасть в глобальный топ-k (слотов всего k).
    merged.sort(key=lambda r: r[0])
    out["latency"]["search_ms"] = (time.perf_counter() - t0) * 1000

    if missing:
        note = (f"коллекции не найдены: {', '.join(missing)} — постройте их "
                f"`python index.py`")
        out["warning"] = (out["warning"] + " | " + note) if out["warning"] else note

    for dist, cid, m in merged[:top_k]:
        rel = max(0.0, min(100.0, (1.0 - dist) * 100.0))   # space=cosine
        out["results"].append({
            "chunk_id": cid,
            "path": m["path"], "name": m["name"], "kind": m["kind"],
            "start": m["start_line"], "end": m["end_line"],
            "code": m["code"], "docstring": m.get("docstring", ""),
            "relevance": rel, "source": m.get("source", ""),
        })

    out["latency"]["total_ms"] = (time.perf_counter() - t_all) * 1000
    return out


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CodeLens search (CLI)")
    ap.add_argument("query", nargs="*", help="текст запроса")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--no-hyde", action="store_true")
    ap.add_argument("--source", choices=list(_SOURCE_COLLECTIONS), default=DEFAULT_SOURCE,
                    help="область поиска: gymhero (офиц.) / all (вся база) / extra")
    args = ap.parse_args()

    query = " ".join(args.query) or "как создаётся токен доступа"
    r = search(query, top_k=args.top_k, use_hyde=not args.no_hyde, source=args.source)

    if r["warning"]:
        print(f"[!] {r['warning']}")
    lat = r["latency"]
    print(f"Запрос: {query!r}  (source={r['source']}, HyDE={'on' if r['hyde_used'] else 'off'}"
          f"{', cache' if r['hyde_from_cache'] else ''})")
    print(f"latency: hyde={lat['hyde_ms']:.0f} ms, embed={lat['embed_ms']:.0f} ms, "
          f"search={lat['search_ms']:.1f} ms, total={lat['total_ms']:.0f} ms")
    print("-" * 78)
    for i, hit in enumerate(r["results"], 1):
        src = f" «{hit['source']}»" if hit.get("source") else ""
        print(f"{i}. [{hit['relevance']:5.1f}%] {hit['chunk_id']}{src}")
        print(f"   {hit['kind']}  {hit['path']}  L{hit['start']}-{hit['end']}")


if __name__ == "__main__":
    _cli()

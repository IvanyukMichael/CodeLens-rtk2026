"""Приоритет 3, п.2: API-эмбеддеры (Cohere embed-v4, Voyage voyage-code-3) на 170.

«Потолок API-моделей» для отчёта (как делал победитель 1 этапа): эмбеддим ТЕ ЖЕ
enriched-тексты чанков gymhero и вопросы облачным API, косинус -> top-5, считаем
P@5 на combined-170 со срезами, сравниваем с локальной bge-m3 enriched.

Важно про методологию: продукт остаётся на воспроизводимой ЛОКАЛЬНОЙ конфигурации
(bge-m3). API-числа идут в отчёт как «потолок, если тестировали», и НЕ становятся
прод-зависимостью от платного ключа.

Ключи — из окружения: COHERE_API_KEY, VOYAGE_API_KEY. Провайдер без ключа
пропускается. Эмбеддинги кэшируются (cache/api_emb_*.pkl) — API не дёргается повторно.

Запуск (GPU не нужен — счёт удалённый; нужны ключи + сеть):
    pip install cohere voyageai
    export COHERE_API_KEY=...     # и/или
    export VOYAGE_API_KEY=...
    python exp_api_embedders.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов

import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map
from score import score_question

REPO = Path("gymhero")
OFFICIAL = Path("eval_questions.json")
EXTENDED = Path("eval/eval_extended.json")
COMBINED = Path("eval/eval_combined.json")
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
ABLATION_FULL = RESULTS_DIR / "ablation_full.csv"


# ─────────────────────────── общие хелперы (как в ablation_full) ───────────────────────────
def load_combined() -> list[dict]:
    if COMBINED.exists():
        return json.loads(COMBINED.read_text(encoding="utf-8"))
    official = [{**q, "subset": "official"} for q in json.loads(OFFICIAL.read_text(encoding="utf-8"))]
    extended = [{**q, "subset": "extended"} for q in json.loads(EXTENDED.read_text(encoding="utf-8"))]
    return official + extended


def aggregate(top5_by_qid: dict, questions: list[dict]) -> dict:
    sc = {q["question_id"]: score_question(top5_by_qid[q["question_id"]], q["correct_chunk_ids"])
          for q in questions}

    def mean(pred):
        v = [sc[q["question_id"]] for q in questions if pred(q)]
        return sum(v) / len(v) if v else float("nan")

    return {
        "p5_total": mean(lambda q: True),
        "p5_easy": mean(lambda q: q["difficulty"] == "easy"),
        "p5_medium": mean(lambda q: q["difficulty"] == "medium"),
        "p5_hard": mean(lambda q: q["difficulty"] == "hard"),
        "p5_ru": mean(lambda q: q["language"] == "ru"),
        "p5_en": mean(lambda q: q["language"] == "en"),
        "p5_official15": mean(lambda q: q.get("subset") == "official"),
        "p5_synthetic": mean(lambda q: q.get("subset") == "extended"),
    }


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norm, 1e-9, None)


def top5_by_cosine(chunk_emb, query_emb, chunk_ids, questions) -> dict:
    chunk_emb, query_emb = _normalize(chunk_emb), _normalize(query_emb)
    out = {}
    for i, q in enumerate(questions):
        sims = chunk_emb @ query_emb[i]
        idx = np.argsort(-sims)[:5]
        out[q["question_id"]] = [chunk_ids[j] for j in idx]
    return out


def _cache_key(model: str, kind: str, texts: list[str]) -> str:
    h = hashlib.sha1((model + "\x00" + kind + "\x00" + "\x00".join(texts)).encode("utf-8"))
    return h.hexdigest()


def _cached_embed(provider_slug, model, kind, texts, embed_fn):
    """Эмбеддинги с дисковым кэшем по содержимому (API не вызывается повторно)."""
    cache_file = CACHE_DIR / f"api_emb_{provider_slug}.pkl"
    cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}
    key = _cache_key(model, kind, texts)
    if key in cache:
        return np.asarray(cache[key], dtype=np.float32)
    vecs = np.asarray(embed_fn(texts, kind), dtype=np.float32)
    cache[key] = vecs
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(cache))
    return vecs


# ─────────────────────────────── провайдеры ───────────────────────────────
def cohere_embedder():
    """Возвращает (model, embed_fn) или None, если нет ключа/SDK."""
    if not os.environ.get("COHERE_API_KEY"):
        return None
    try:
        import cohere
    except ImportError:
        print("Cohere: нет пакета — `pip install cohere`. Пропуск.")
        return None
    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    model = "embed-v4.0"
    itype = {"doc": "search_document", "query": "search_query"}

    def embed(texts, kind):
        out = []
        for s in range(0, len(texts), 96):                # лимит батча Cohere
            resp = co.embed(texts=texts[s:s + 96], model=model,
                            input_type=itype[kind], embedding_types=["float"])
            emb = resp.embeddings
            out.extend(getattr(emb, "float", None) or getattr(emb, "float_"))
        return out

    return model, embed


def voyage_embedder():
    if not os.environ.get("VOYAGE_API_KEY"):
        return None
    try:
        import voyageai
    except ImportError:
        print("Voyage: нет пакета — `pip install voyageai`. Пропуск.")
        return None
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    model = "voyage-code-3"
    itype = {"doc": "document", "query": "query"}

    def embed(texts, kind):
        out = []
        for s in range(0, len(texts), 128):               # лимит батча Voyage
            res = vo.embed(texts[s:s + 128], model=model, input_type=itype[kind])
            out.extend(res.embeddings)
        return out

    return model, embed


PROVIDERS = {"cohere": cohere_embedder, "voyage": voyage_embedder}


def _baseline_row() -> dict | None:
    """Строка bge-m3 + enriched из results/ablation_full.csv (как референс), если есть."""
    if not ABLATION_FULL.exists():
        return None
    with open(ABLATION_FULL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("config") == "bge-m3 + enriched":
                return {"config": "bge-m3 + enriched (локально, референс)",
                        **{k: float(v) for k, v in r.items()
                           if k.startswith("p5_") and v not in ("", None)}}
    return None


def main():
    questions = load_combined()
    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    enriched_map = build_enriched_map(chunks)
    docs = [enriched_map[cid] for cid in chunk_ids]
    queries = [q["query"] for q in questions]
    print(f"вопросов: {len(questions)}  |  чанков gymhero: {len(chunks)}")

    rows = []
    ref = _baseline_row()
    if ref:
        rows.append(ref)

    any_run = False
    for name, factory in PROVIDERS.items():
        prov = factory()
        if prov is None:
            print(f"[{name}] пропуск (нет ключа {name.upper()}_API_KEY или SDK).")
            continue
        model, embed = prov
        print(f"[{name}] {model}: эмбеддинг {len(docs)} чанков + {len(queries)} вопросов ...")
        chunk_emb = _cached_embed(name, model, "doc", docs, embed)
        query_emb = _cached_embed(name, model, "query", queries, embed)
        top5 = top5_by_cosine(chunk_emb, query_emb, chunk_ids, questions)
        rows.append({"config": f"{name} {model}", **aggregate(top5, questions)})
        any_run = True

    if not any_run:
        print("\nНи один API-провайдер не запущен. Задай COHERE_API_KEY и/или VOYAGE_API_KEY.")
        if not rows:
            return

    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en",
            "p5_official15", "p5_synthetic"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ablation_api_embedders.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 108)
    print(f"{'config':<40}{'P@5':>7}{'easy':>7}{'med':>7}{'hard':>7}{'ru':>7}{'en':>7}"
          f"{'offic':>8}{'ext':>8}")
    print("-" * 108)
    for r in rows:
        print(f"{r['config']:<40}{fmt(r.get('p5_total')):>7}{fmt(r.get('p5_easy')):>7}"
              f"{fmt(r.get('p5_medium')):>7}{fmt(r.get('p5_hard')):>7}{fmt(r.get('p5_ru')):>7}"
              f"{fmt(r.get('p5_en')):>7}{fmt(r.get('p5_official15')):>8}{fmt(r.get('p5_synthetic')):>8}")
    print("=" * 108)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

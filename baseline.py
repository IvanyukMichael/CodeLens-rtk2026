"""CodeLens RAG — baseline-ретривер.

Самая простая честная конфигурация (точка отсчёта для последующих ablation):
  чанки из gymhero/  →  эмбеддинг СЫРОГО кода чанка моделью
  paraphrase-multilingual-MiniLM-L12-v2  →  для каждого вопроса топ-5 по
  косинусному сходству (numpy)  →  results.json  →  score.py.

Никакого enrichment / reranking / HyDE / гибрида здесь нет намеренно — это
baseline, относительно которого мы будем измерять каждую оптимизацию.

Запуск:
    python baseline.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from extract_chunks import extract_repo

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REPO = Path("gymhero")
EVAL = Path("eval_questions.json")
RESULTS = Path("results.json")
CACHE = Path("cache/baseline_embeddings.pkl")


def build_doc(chunk: dict) -> str:
    """Текст, который эмбеддим для чанка. Baseline = сырой код."""
    return chunk["code"]


def load_or_build_embeddings(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    """Возвращает матрицу нормированных эмбеддингов (C, D), кэшируя на диск.

    Кэш инвалидируется, если поменялась модель или набор chunk_id."""
    chunk_ids = [c["chunk_id"] for c in chunks]

    if CACHE.exists():
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("model") == MODEL_NAME and cached.get("chunk_ids") == chunk_ids:
            print(f"Эмбеддинги загружены из кэша: {CACHE}")
            return np.asarray(cached["embeddings"], dtype=np.float32)
        print("Кэш устарел (изменились чанки или модель) — пересчитываю.")

    docs = [build_doc(c) for c in chunks]
    emb = model.encode(
        docs, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    emb = np.asarray(emb, dtype=np.float32)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(
            {"model": MODEL_NAME, "chunk_ids": chunk_ids, "embeddings": emb}, f
        )
    print(f"Эмбеддинги посчитаны и сохранены в кэш: {CACHE}")
    return emb


def main() -> None:
    print("[1/5] Извлекаю чанки из", REPO, "...")
    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"      чанков: {len(chunks)}")

    print(f"[2/5] Загружаю модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("[3/5] Эмбеддинги чанков ...")
    chunk_emb = load_or_build_embeddings(model, chunks)  # (C, D), нормированы

    print("[4/5] Кодирую вопросы и ищу топ-5 (косинус через numpy) ...")
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    queries = [q["query"] for q in questions]
    q_emb = np.asarray(
        model.encode(queries, normalize_embeddings=True), dtype=np.float32
    )  # (Q, D), нормированы

    results = []
    latencies_ms = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        sims = chunk_emb @ q_emb[i]          # косинус (всё нормировано) -> (C,)
        top5_idx = np.argpartition(-sims, 5)[:5]
        top5_idx = top5_idx[np.argsort(-sims[top5_idx])]
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        results.append({
            "question_id": q["question_id"],
            "top_5_chunks": [chunk_ids[j] for j in top5_idx],
        })

    RESULTS.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    print(f"      {RESULTS} записан ({len(results)} вопросов); "
          f"поиск (без кодирования) p50={p50:.2f} мс, p95={p95:.2f} мс")

    print("[5/5] Запускаю score.py ...\n")
    proc = subprocess.run(
        [sys.executable, "score.py",
         "--predictions", str(RESULTS),
         "--questions", str(EVAL)],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    total = next(
        (ln.split(":", 1)[1].strip()
         for ln in proc.stdout.splitlines() if ln.startswith("Total score:")),
        "n/a",
    )
    print("\n" + "=" * 40)
    print(f"BASELINE Total score (Precision@5): {total}")
    print("=" * 40)


if __name__ == "__main__":
    main()

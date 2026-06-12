"""HyDE поверх базы bge-m3 + enriched (P@5=0.767).

Идея HyDE: на запрос (RU/EN) генерим ГИПОТЕТИЧЕСКИЙ Python-код локальной LLM
(Ollama qwen2.5-coder:7b), эмбеддим его той же bge-m3 и ищем по нему — это
сближает «запрос на естественном языке» с «кодом» в одном (кодовом) пространстве.

Режимы:
  base       — поиск по эмбеддингу запроса (это и есть наша база 0.767);
  hyde_pure  — поиск только по эмбеддингу сгенерированного кода;
  hyde_mix   — поиск по усреднённому эмбеддингу (запрос + HyDE), равные веса.

Замечание: усреднение эмбеддингов == равновесная фьюзия скоров (ранжирование
инвариантно к ренормировке суммы), поэтому mix считаем как qvec = q + hyde.

Генерации LLM кэшируются в cache/hyde_generations.json (медленно — не повторять).

Запуск:
    python hyde.py                 # сгенерировать (если нужно) + прогнать ablation
    python hyde.py --regen         # принудительно перегенерировать все ответы LLM
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import requests

from extract_chunks import extract_repo
from eval_runner import (
    EVAL, REPO, RESULTS_DIR, CACHE_DIR,
    get_embeddings, run_score, _make_loader,
)

RETRIEVER = "BAAI/bge-m3"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5-coder:7b"
HYDE_CACHE = CACHE_DIR / "hyde_generations.json"

SYSTEM_PROMPT = (
    "You are a senior Python backend engineer. You are given a question "
    "(in Russian or English) about a Python web/backend codebase. "
    "Write ONE short, plausible Python function or class that implements or "
    "answers what the question is about.\n"
    "Rules:\n"
    "- Use idiomatic ENGLISH identifiers (function/class/variable names) as in real "
    "code, even when the question is in Russian.\n"
    "- Include a realistic signature, a one-line docstring, and a body.\n"
    "- Output ONLY raw Python code: no markdown fences, no prose, no explanations."
)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def ollama_generate(question: str, temperature: float = 0.0) -> str:
    """Генерирует HyDE-фрагмент локальной LLM.

    temperature=0 (по умолчанию) — greedy-декодирование: генерация
    детерминирована, поэтому «живьём» совпадает с кэшем и вопрос
    воспроизводимости снимается (см. эксперимент exp_hyde_temp0.py)."""
    r = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": question,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 400},
    }, timeout=180)
    r.raise_for_status()
    text = r.json()["response"].strip()
    return _FENCE.sub("", text).strip()


def load_or_generate(questions, regen=False) -> dict:
    cache = {}
    if HYDE_CACHE.exists() and not regen:
        cache = json.loads(HYDE_CACHE.read_text(encoding="utf-8"))

    missing = [q for q in questions if q["question_id"] not in cache]
    if missing:
        print(f"Генерация HyDE для {len(missing)} вопросов через {LLM_MODEL} ...")
        for q in missing:
            code = ollama_generate(q["query"])
            cache[q["question_id"]] = code
            print(f"  {q['question_id']} [{q['language']}] -> {len(code)} симв.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        HYDE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    else:
        print(f"HyDE-генерации загружены из кэша: {HYDE_CACHE}")
    return cache


def preds_from_vecs(qvecs, chunk_emb, chunk_ids, questions, k=5):
    res = []
    for i, q in enumerate(questions):
        sims = chunk_emb @ qvecs[i]
        idx = np.argsort(-sims)[:k]
        res.append({"question_id": q["question_id"],
                    "top_5_chunks": [chunk_ids[j] for j in idx]})
    return res


def score_vecs(qvecs, chunk_emb, chunk_ids, questions, tag):
    preds = preds_from_vecs(qvecs, chunk_emb, chunk_ids, questions)
    path = RESULTS_DIR / f"preds_bge-m3_enriched_{tag}.json"
    path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_score(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    questions = json.loads(EVAL.read_text(encoding="utf-8"))

    hyde_map = load_or_generate(questions, regen=args.regen)

    # пример сгенерированного кода (русский вопрос)
    ru = next((q for q in questions if q["language"] == "ru"), questions[0])
    print(f"\n--- пример HyDE ({ru['question_id']}, ru) ---")
    print("Q:", ru["query"])
    print(hyde_map[ru["question_id"]][:400])

    # bge-m3 enriched эмбеддинги чанков + запросов (из кэша)
    get_model, holder = _make_loader(RETRIEVER, device)
    chunk_emb, query_emb, hit = get_embeddings(RETRIEVER, True, chunks, questions, get_model)
    print(f"\nbge-m3 enriched эмбеддинги: cache_hit={hit}")

    # эмбеддинг HyDE-кода той же bge-m3
    model = get_model()
    hyde_codes = [hyde_map[q["question_id"]] for q in questions]
    hyde_emb = np.asarray(
        model.encode(hyde_codes, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = score_vecs(query_emb, chunk_emb, chunk_ids, questions, "base")
    pure = score_vecs(hyde_emb, chunk_emb, chunk_ids, questions, "hyde_pure")
    mix = score_vecs(query_emb + hyde_emb, chunk_emb, chunk_ids, questions, "hyde_mix")

    rows = [
        {"config": "bge-m3+enriched", **base},
        {"config": "bge-m3+enriched+hyde_pure", **pure},
        {"config": "bge-m3+enriched+hyde_mix", **mix},
    ]
    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en"]
    csv_path = RESULTS_DIR / "ablation_hyde.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 86)
    print(f"{'config':<28} {'P@5':>7} {'easy':>7} {'med':>7} {'hard':>7} {'ru':>7} {'en':>7}")
    print("-" * 86)
    for r in rows:
        print(f"{r['config']:<28} {fmt(r['p5_total']):>7} {fmt(r['p5_easy']):>7} "
              f"{fmt(r['p5_medium']):>7} {fmt(r['p5_hard']):>7} "
              f"{fmt(r['p5_ru']):>7} {fmt(r['p5_en']):>7}")
    print("=" * 86)

    for tag, r in (("hyde_pure", pure), ("hyde_mix", mix)):
        print(f"\nДельта {tag} - база:")
        print(f"  P@5 total : {r['p5_total'] - base['p5_total']:+.3f}  "
              f"({base['p5_total']:.3f} -> {r['p5_total']:.3f})")
        print(f"  hard      : {r['p5_hard'] - base['p5_hard']:+.3f}  "
              f"({base['p5_hard']:.3f} -> {r['p5_hard']:.3f})")
        print(f"  en        : {r['p5_en'] - base['p5_en']:+.3f}  "
              f"({base['p5_en']:.3f} -> {r['p5_en']:.3f})")

    # короткий α-свип для mix (qvec = a*query + (1-a)*hyde); ранжирование инвариантно к масштабу
    print("\nα-свип mix (a*query + (1-a)*hyde):")
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        qv = a * query_emb + (1 - a) * hyde_emb
        sc = score_vecs(qv, chunk_emb, chunk_ids, questions, "tmp_sweep")
        print(f"  a={a:.1f}  P@5={sc['p5_total']:.3f}  hard={sc['p5_hard']:.3f}  en={sc['p5_en']:.3f}")
    (RESULTS_DIR / "preds_bge-m3_enriched_tmp_sweep.json").unlink(missing_ok=True)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

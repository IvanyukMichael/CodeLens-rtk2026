"""Эксперимент: детерминизм HyDE через temperature=0.

Цель — проверить, снимает ли greedy-декодирование (temp=0) стохастику Ollama,
чтобы живая генерация совпадала с кэшем и вопрос «кэш vs живьё» отпал.

На объединённом eval (71 вопрос) считаем P@5 (тот же score_question) для 4 строк
на ОДНОМ коде (bge-m3 + enriched, mix qvec=0.5*q+0.5*hyde), БЕЗ продакшн-сид-кэша
search.py — генерим напрямую через Ollama в отдельные cache/hyde_temp0_*.json:

  A) temp=0.2 seed-cache (frozen)  — замороженные генерации экспериментов;
  B) temp=0.2 live                 — живая генерация при temp=0.2 (стохастика);
  C) temp=0 live (run 1)           — живая генерация greedy;
  D) temp=0 live (run 2)           — повтор greedy (проверка детерминизма).

Детерминизм: сравниваем текст генераций и P@5 между run1 и run2.

Запуск:
    python exp_hyde_temp0.py
"""

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов HF

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import hyde
from extract_chunks import extract_repo
from enrich import build_enriched_map
from score import score_question

REPO = Path("gymhero")
COMBINED = Path("eval/eval_combined.json")
CACHE = Path("cache")
BGE = "BAAI/bge-m3"


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def gen_set(questions, temperature, label):
    """Живая генерация HyDE для всех вопросов (прямой Ollama, без кэша search.py)."""
    out = {}
    t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        out[q["question_id"]] = hyde.ollama_generate(q["query"], temperature=temperature)
        if i % 10 == 0:
            print(f"  [{label}] {i}/{len(questions)}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    print(f"  [{label}] готово: {len(questions)} генераций за "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return out


def p5_per_q(gens, questions, model, chunk_emb, chunk_ids, q_emb):
    """{qid: P@5} для набора генераций (mix qvec=0.5*q+0.5*hyde)."""
    h = np.asarray(
        model.encode([gens[q["question_id"]] for q in questions],
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    per = {}
    for i, q in enumerate(questions):
        qmix = 0.5 * q_emb[i] + 0.5 * h[i]
        sims = chunk_emb @ qmix
        top5 = [chunk_ids[j] for j in np.argsort(-sims)[:5]]
        per[q["question_id"]] = score_question(top5, q["correct_chunk_ids"])
    return per


def agg(per, questions):
    def m(pred):
        v = [per[q["question_id"]] for q in questions if pred(q)]
        return sum(v) / len(v) if v else float("nan")
    return dict(total=m(lambda q: True),
                official=m(lambda q: q.get("subset") == "official"),
                extended=m(lambda q: q.get("subset") == "extended"))


def main():
    dev = device()
    print(f"device: {dev}")
    model = SentenceTransformer(BGE, device=dev)

    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    enr = build_enriched_map(chunks)
    chunk_emb = np.asarray(
        model.encode([enr[c["chunk_id"]] for c in chunks],
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    questions = json.loads(COMBINED.read_text(encoding="utf-8"))
    q_emb = np.asarray(
        model.encode([q["query"] for q in questions],
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    print(f"chunks={len(chunks)}  questions={len(questions)} "
          f"(official={sum(q.get('subset')=='official' for q in questions)}, "
          f"extended={sum(q.get('subset')=='extended' for q in questions)})", flush=True)

    rows = {}

    # A) temp=0.2 seed-cache (frozen генерации экспериментов)
    frozen = json.loads((CACHE / "hyde_generations.json").read_text(encoding="utf-8"))
    rows["temp=0.2 seed-cache (frozen)"] = agg(
        p5_per_q(frozen, questions, model, chunk_emb, chunk_ids, q_emb), questions)
    print("[A] seed-cache посчитан", flush=True)

    # B) temp=0.2 live
    print("[B] генерация temp=0.2 live ...", flush=True)
    g_b = gen_set(questions, 0.2, "temp0.2 live")
    (CACHE / "hyde_temp02_live.json").write_text(
        json.dumps(g_b, ensure_ascii=False, indent=2), encoding="utf-8")
    rows["temp=0.2 live"] = agg(
        p5_per_q(g_b, questions, model, chunk_emb, chunk_ids, q_emb), questions)

    # C) temp=0 live run1
    print("[C] генерация temp=0 live (run 1) ...", flush=True)
    g_c = gen_set(questions, 0.0, "temp0 run1")
    (CACHE / "hyde_temp0_run1.json").write_text(
        json.dumps(g_c, ensure_ascii=False, indent=2), encoding="utf-8")
    per_c = p5_per_q(g_c, questions, model, chunk_emb, chunk_ids, q_emb)
    rows["temp=0 live (run 1)"] = agg(per_c, questions)

    # D) temp=0 live run2
    print("[D] генерация temp=0 live (run 2) ...", flush=True)
    g_d = gen_set(questions, 0.0, "temp0 run2")
    (CACHE / "hyde_temp0_run2.json").write_text(
        json.dumps(g_d, ensure_ascii=False, indent=2), encoding="utf-8")
    per_d = p5_per_q(g_d, questions, model, chunk_emb, chunk_ids, q_emb)
    rows["temp=0 live (run 2)"] = agg(per_d, questions)

    # --- детерминизм ---
    diff_text = [qid for qid in g_c if g_c[qid] != g_d[qid]]
    diff_score = [q["question_id"] for q in questions
                  if per_c[q["question_id"]] != per_d[q["question_id"]]]

    # --- таблица ---
    order = ["temp=0.2 seed-cache (frozen)", "temp=0.2 live",
             "temp=0 live (run 1)", "temp=0 live (run 2)"]
    print("\n" + "=" * 62)
    print(f"{'config':<32}{'P@5(71)':>9}{'offic15':>9}{'ext56':>8}")
    print("-" * 62)
    for name in order:
        r = rows[name]
        print(f"{name:<32}{r['total']:>9.3f}{r['official']:>9.3f}{r['extended']:>8.3f}")
    print("=" * 62)

    print(f"\nДетерминизм temp=0 (run1 vs run2):")
    print(f"  текст генераций идентичен: {len(questions) - len(diff_text)}/{len(questions)} "
          f"(различий: {len(diff_text)})")
    print(f"  P@5 run1 == run2: {'ДА' if not diff_score else 'НЕТ — ' + str(diff_score)}")
    if diff_text:
        print(f"  вопросы с разным текстом: {diff_text[:12]}")

    out = {"rows": rows,
           "determinism": {"text_diff_qids": diff_text, "score_diff_qids": diff_score,
                           "n_questions": len(questions)}}
    (Path("results") / "exp_hyde_temp0.json").parent.mkdir(exist_ok=True)
    Path("results/exp_hyde_temp0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nJSON: results/exp_hyde_temp0.json")


if __name__ == "__main__":
    main()

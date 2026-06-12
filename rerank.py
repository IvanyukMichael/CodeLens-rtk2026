"""Cross-encoder reranking поверх базы bge-m3 + enriched (P@5=0.767).

Пайплайн:
    bge-m3(enriched) -> top-20 кандидатов  (НЕ top-5!)
        -> cross-encoder BAAI/bge-reranker-v2-m3 по парам (запрос, enriched-текст чанка)
        -> сортировка по score -> новый top-5

Сравнивает базу и базу+rerank, пишет results/ablation_rerank.csv, отдельно
показывает дельту по целевым срезам hard и en.

Скоры реранкера кэшируются на уровне пар (qid, chunk_id) в
cache/rerank_scores_{reranker_slug}.pkl — повторный прогон не пересчитывает.

Запуск:
    python rerank.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map
from eval_runner import (
    EVAL, REPO, RESULTS_DIR, CACHE_DIR,
    get_embeddings, retrieve_top5, run_score, slug, _make_loader,
)

RETRIEVER = "BAAI/bge-m3"
ENRICHED = True
RERANKER = "BAAI/bge-reranker-v2-m3"
TOP_N = 20          # кандидатов от ретривера на вход реранкеру
TOP_K = 5           # финальный размер выдачи


def top_n_candidates(chunk_emb, query_emb, chunk_ids, n):
    """Список из top-n chunk_id на каждый вопрос (по косинусу, всё нормировано)."""
    out = []
    for i in range(len(query_emb)):
        sims = chunk_emb @ query_emb[i]
        idx = np.argsort(-sims)[:n]
        out.append([chunk_ids[j] for j in idx])
    return out


def rerank(ce, questions, candidates, enriched_map, score_cache, k=TOP_K):
    """Переранжировать кандидатов cross-encoder'ом, вернуть top-k chunk_id на вопрос.

    score_cache: dict {(qid, chunk_id): float} — мутируется (дополняется)."""
    # 1) собрать пары, которых ещё нет в кэше
    todo = []
    for q, cands in zip(questions, candidates):
        for cid in cands:
            if (q["question_id"], cid) not in score_cache:
                todo.append((q["question_id"], cid, q["query"], enriched_map[cid]))

    if todo:
        pairs = [[query, doc] for (_, _, query, doc) in todo]
        scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        for (qid, cid, _, _), s in zip(todo, scores):
            score_cache[(qid, cid)] = float(s)

    # 2) пересортировать кандидатов по score реранкера
    results = []
    for q, cands in zip(questions, candidates):
        ranked = sorted(cands, key=lambda cid: score_cache[(q["question_id"], cid)],
                        reverse=True)
        results.append({"question_id": q["question_id"], "top_5_chunks": ranked[:k]})
    return results


def recall_at_n(candidates, questions, n):
    """Доля правильных чанков, попавших в top-n кандидатов (потолок реранкинга)."""
    from score import chunks_match
    hit = tot = 0
    for q, cands in zip(questions, candidates):
        for ref in q["correct_chunk_ids"]:
            tot += 1
            if any(chunks_match(c, ref) for c in cands[:n]):
                hit += 1
    return hit / tot if tot else float("nan")


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    enriched_map = build_enriched_map(chunks)

    # --- база: эмбеддинги bge-m3 enriched (из кэша; модель грузится только при промахе) ---
    get_model, holder = _make_loader(RETRIEVER, device)
    chunk_emb, query_emb, hit = get_embeddings(
        RETRIEVER, ENRICHED, chunks, questions, get_model)
    print(f"bge-m3 enriched эмбеддинги: cache_hit={hit}")
    if holder["m"] is not None:                       # освободить ретривер, если грузили
        del holder["m"]; holder["m"] = None
        torch.cuda.empty_cache()

    # --- база top-5 ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_results = retrieve_top5(chunk_emb, query_emb, chunk_ids, questions, k=TOP_K)
    base_pred = RESULTS_DIR / "preds_bge-m3_enriched.json"
    base_pred.write_text(json.dumps(base_results, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    base = run_score(base_pred)

    # --- кандидаты top-20 + потолок (recall@20) ---
    candidates = top_n_candidates(chunk_emb, query_emb, chunk_ids, TOP_N)
    print(f"recall@5 (база)   = {recall_at_n(candidates, questions, TOP_K):.3f}")
    print(f"recall@{TOP_N} (потолок) = {recall_at_n(candidates, questions, TOP_N):.3f}")

    # --- реранкинг ---
    cache_file = CACHE_DIR / f"rerank_scores_{slug(RERANKER)}.pkl"
    score_cache = {}
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            score_cache = pickle.load(f)

    from sentence_transformers import CrossEncoder
    print(f"загрузка реранкера {RERANKER} ...")
    ce = CrossEncoder(RERANKER, device=device, max_length=512)

    rr_results = rerank(ce, questions, candidates, enriched_map, score_cache)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(score_cache, f)

    rr_pred = RESULTS_DIR / "preds_bge-m3_enriched_rerank.json"
    rr_pred.write_text(json.dumps(rr_results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    rr = run_score(rr_pred)

    # --- CSV ---
    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en"]
    rows = [
        {"config": "bge-m3+enriched", **base},
        {"config": "bge-m3+enriched+rerank", **rr},
    ]
    csv_path = RESULTS_DIR / "ablation_rerank.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- вывод ---
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 84)
    print(f"{'config':<26} {'P@5':>7} {'easy':>7} {'med':>7} {'hard':>7} {'ru':>7} {'en':>7}")
    print("-" * 84)
    for r in rows:
        print(f"{r['config']:<26} {fmt(r['p5_total']):>7} {fmt(r['p5_easy']):>7} "
              f"{fmt(r['p5_medium']):>7} {fmt(r['p5_hard']):>7} "
              f"{fmt(r['p5_ru']):>7} {fmt(r['p5_en']):>7}")
    print("=" * 84)

    def d(key):
        return rr[key] - base[key]
    print("\nДельта (rerank - база):")
    print(f"  P@5 total : {d('p5_total'):+.3f}  ({base['p5_total']:.3f} -> {rr['p5_total']:.3f})")
    print("  --- целевая зона ---")
    print(f"  hard      : {d('p5_hard'):+.3f}  ({base['p5_hard']:.3f} -> {rr['p5_hard']:.3f})")
    print(f"  en        : {d('p5_en'):+.3f}  ({base['p5_en']:.3f} -> {rr['p5_en']:.3f})")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

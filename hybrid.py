"""Гибридный поиск (dense + BM25 через RRF) поверх базы
bge-m3 + enriched + HyDE-mix (P@5=0.822).

dense  = HyDE-mix вектор (query_emb + hyde_emb), косинус к enriched-эмбеддингам чанков.
BM25   = rank-bm25 по enriched-тексту чанков; токенизация бьёт snake_case/camelCase
         на слова (важно, чтобы матчить имена функций: create_access_token ->
         create, access, token).
fusion = RRF: score(c) = Σ_source 1/(k + rank_source(c)), k=60, top-N=30 с каждой стороны.

Два варианта токенизации ЗАПРОСА для BM25:
  q      — только текст вопроса (как есть; русский остаётся русским);
  q+hyde — вопрос + сгенерированный HyDE-код (в нём английские имена -> матч с кодом).

Запуск:
    python hybrid.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import hyde
from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map
from eval_runner import EVAL, REPO, RESULTS_DIR, get_embeddings, run_score, _make_loader
from score import score_question

RETRIEVER = "BAAI/bge-m3"
TOP_N = 30          # кандидатов с каждой стороны в RRF
RRF_K = 60
TOP_K = 5

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """snake_case + camelCase -> отдельные слова, lower; одиночные буквы выбрасываем."""
    out = []
    for w in _WORD.findall(text):              # подчёркивания/точки уже разделители
        for sub in _CAMEL.sub(" ", w).split():
            sub = sub.lower()
            if len(sub) >= 2 or sub.isdigit():
                out.append(sub)
    return out


def dense_ranking(chunk_emb, qvec, chunk_ids):
    order = np.argsort(-(chunk_emb @ qvec), kind="stable")
    return [chunk_ids[i] for i in order]


def bm25_ranking(bm25, chunk_ids, query_tokens):
    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores, kind="stable")
    return [(chunk_ids[i], float(scores[i])) for i in order]


def rrf(dense_ids, bm25_scored, w_b=1.0, n=TOP_N, k=RRF_K, topk=TOP_K):
    """Взвешенное RRF: score(c) = 1/(k+rank_dense) + w_b/(k+rank_bm25).
    w_b=1.0 — классический RRF; w_b<1 приглушает BM25. BM25-кандидаты с нулевым
    скором не вносятся (это не «попадание»)."""
    agg = defaultdict(float)
    for rank, cid in enumerate(dense_ids[:n], start=1):
        agg[cid] += 1.0 / (k + rank)
    rank = 0
    for cid, sc in bm25_scored:
        if sc <= 0:
            continue
        rank += 1
        if rank > n:
            break
        agg[cid] += w_b * (1.0 / (k + rank))
    return sorted(agg, key=lambda c: agg[c], reverse=True)[:topk]


def evaluate(top5_by_q, questions, tag):
    preds = [{"question_id": q["question_id"], "top_5_chunks": top5_by_q[q["question_id"]]}
             for q in questions]
    path = RESULTS_DIR / f"preds_hybrid_{tag}.json"
    path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_score(path)


def per_question(top5_by_q, questions):
    return {q["question_id"]: score_question(top5_by_q[q["question_id"]], q["correct_chunk_ids"])
            for q in questions}


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    enriched_map = build_enriched_map(chunks)
    hyde_map = hyde.load_or_generate(questions)

    # --- dense: bge-m3 enriched (кэш) + HyDE-эмбеддинг -> mix ---
    get_model, _ = _make_loader(RETRIEVER, device)
    chunk_emb, query_emb, hit = get_embeddings(RETRIEVER, True, chunks, questions, get_model)
    print(f"bge-m3 enriched эмбеддинги: cache_hit={hit}")
    model = get_model()
    hyde_emb = np.asarray(
        model.encode([hyde_map[q["question_id"]] for q in questions],
                     normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    qmix = query_emb + hyde_emb                 # равновесная HyDE-mix

    # --- BM25 по enriched-тексту чанков ---
    corpus_tokens = [tokenize(enriched_map[cid]) for cid in chunk_ids]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"BM25 индекс: {len(corpus_tokens)} чанков, "
          f"средняя длина {np.mean([len(t) for t in corpus_tokens]):.0f} токенов")

    W_TUNED = 0.2          # подобрано на этом же eval (см. предупреждение об оверфите ниже)
    base_top5, hyb_q, hyb_qh, hyb_qh_t = {}, {}, {}, {}
    for i, q in enumerate(questions):
        qid = q["question_id"]
        d_ids = dense_ranking(chunk_emb, qmix[i], chunk_ids)
        base_top5[qid] = d_ids[:TOP_K]
        bm_q = bm25_ranking(bm25, chunk_ids, tokenize(q["query"]))
        bm_qh = bm25_ranking(bm25, chunk_ids, tokenize(q["query"] + " " + hyde_map[qid]))
        hyb_q[qid] = rrf(d_ids, bm_q, w_b=1.0)
        hyb_qh[qid] = rrf(d_ids, bm_qh, w_b=1.0)
        hyb_qh_t[qid] = rrf(d_ids, bm_qh, w_b=W_TUNED)

    base = evaluate(base_top5, questions, "base")
    hq = evaluate(hyb_q, questions, "rrf_q")
    hqh = evaluate(hyb_qh, questions, "rrf_q_hyde")
    hqt = evaluate(hyb_qh_t, questions, "rrf_q_hyde_w02")

    rows = [
        {"config": "bge-m3+enriched+hyde_mix", **base},
        {"config": "+hybrid_rrf (q, w=1)", **hq},
        {"config": "+hybrid_rrf (q+hyde, w=1)", **hqh},
        {"config": "+hybrid_rrf (q+hyde, w=0.2)", **hqt},
    ]
    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en"]
    csv_path = RESULTS_DIR / "ablation_hybrid.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 88)
    print(f"{'config':<28} {'P@5':>7} {'easy':>7} {'med':>7} {'hard':>7} {'ru':>7} {'en':>7}")
    print("-" * 88)
    for r in rows:
        print(f"{r['config']:<28} {fmt(r['p5_total']):>7} {fmt(r['p5_easy']):>7} "
              f"{fmt(r['p5_medium']):>7} {fmt(r['p5_hard']):>7} "
              f"{fmt(r['p5_ru']):>7} {fmt(r['p5_en']):>7}")
    print("=" * 88)

    for tag, r in (("rrf q (w=1)", hq), ("rrf q+hyde (w=1)", hqh), ("rrf q+hyde (w=0.2)", hqt)):
        print(f"\nДельта {tag} - dense:")
        print(f"  P@5={r['p5_total']-base['p5_total']:+.3f}  "
              f"hard={r['p5_hard']-base['p5_hard']:+.3f}  "
              f"en={r['p5_en']-base['p5_en']:+.3f}  "
              f"ru={r['p5_ru']-base['p5_ru']:+.3f}")
    print("\n⚠ w=0.2 подобран на этих же 15 вопросах — выигрыш +0.011 в пределах шума "
          "(1 вопрос ≈ 0.067). Робастный выбор для прод — чистый dense HyDE-mix.")

    # --- по-вопросный разбор: где гибрид помог/навредил vs чистый dense ---
    sd = per_question(base_top5, questions)
    s1 = per_question(hyb_qh, questions)        # q+hyde, w=1
    s2 = per_question(hyb_qh_t, questions)      # q+hyde, w=0.2 (лучший)
    print("\nПо вопросам (dense -> rrf_w1 / rrf_w0.2, токены q+hyde), * = изменение vs dense:")
    print(f"{'qid':<6}{'lang':<5}{'diff':<8}{'dense':>7}{'rrf_w1':>8}{'rrf_w02':>9}  note")
    for q in questions:
        qid = q["question_id"]
        d, a, b = sd[qid], s1[qid], s2[qid]
        mark = ""
        if b > d:
            mark = "* helped"
        elif b < d:
            mark = "* hurt"
        print(f"{qid:<6}{q['language']:<5}{q['difficulty']:<8}"
              f"{d:>7.2f}{a:>8.2f}{b:>9.2f}  {mark}")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

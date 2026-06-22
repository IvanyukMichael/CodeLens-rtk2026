"""Полная ablation-цепочка на объединённом eval (15 официальных + N расширенных).

База — gymhero (extract_repo, 155 чанков): harness строит эмбеддинги напрямую,
НЕ читает ChromaDB, поэтому числа всегда на изолированной gymhero-коллекции и не
зависят от размера общей базы (см. REPORT §2.5, §3). Авторитетна колонка
p5_official15; combined-набор (p5_total) — проверка устойчивости (в нём 254
LLM-сгенерированных вопроса искажают часть сигналов, см. REPORT §3.4/§4).

Считаем per-question Precision@5 (через score.score_question — тот же скорер) для
ключевых конфигураций, агрегируем по срезам (easy/medium/hard, ru/en) и по
подмножествам (official vs extended), пишем results/ablation_full.csv. Затем —
парный бутстрап значимости ключевых пар.

Запуск:
    python ablation_full.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import hyde
import hybrid
import rerank
from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map
from eval_runner import REPO, RESULTS_DIR, CACHE_DIR, get_embeddings, _make_loader, _free, slug
from score import score_question

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE = "BAAI/bge-m3"
OFFICIAL = Path("eval_questions.json")
EXTENDED = Path("eval/eval_extended.json")
COMBINED = Path("eval/eval_combined.json")


def load_combined():
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    extended = json.loads(EXTENDED.read_text(encoding="utf-8"))
    for q in official:
        q["subset"] = "official"
    for q in extended:
        q["subset"] = "extended"
    combined = official + extended
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    COMBINED.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return combined


def top5_by_vecs(chunk_emb, qvecs, chunk_ids, questions):
    out = {}
    for i, q in enumerate(questions):
        sims = chunk_emb @ qvecs[i]
        idx = np.argpartition(-sims, 5)[:5]
        idx = idx[np.argsort(-sims[idx])]
        out[q["question_id"]] = [chunk_ids[j] for j in idx]
    return out


def aggregate(top5_by_qid, questions):
    sc = {q["question_id"]: score_question(top5_by_qid[q["question_id"]], q["correct_chunk_ids"])
          for q in questions}

    def mean(pred):
        v = [sc[q["question_id"]] for q in questions if pred(q)]
        return sum(v) / len(v) if v else float("nan")

    row = {
        "p5_total": mean(lambda q: True),
        "p5_easy": mean(lambda q: q["difficulty"] == "easy"),
        "p5_medium": mean(lambda q: q["difficulty"] == "medium"),
        "p5_hard": mean(lambda q: q["difficulty"] == "hard"),
        "p5_ru": mean(lambda q: q["language"] == "ru"),
        "p5_en": mean(lambda q: q["language"] == "en"),
        "p5_official15": mean(lambda q: q["subset"] == "official"),
        "p5_extended": mean(lambda q: q["subset"] == "extended"),
    }
    return row, sc


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    questions = load_combined()
    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    enriched_map = build_enriched_map(chunks)
    print(f"вопросов: {len(questions)} (official={sum(q['subset']=='official' for q in questions)}, "
          f"extended={sum(q['subset']=='extended' for q in questions)})")

    top5 = {}    # config -> {qid: top5}

    # === 1. baseline: MiniLM raw ===
    print("\n[1] MiniLM raw ...")
    gm, h = _make_loader(MINILM, device)
    ce_m, qe_m, _ = get_embeddings(MINILM, False, chunks, questions, gm)
    top5["baseline (MiniLM raw)"] = top5_by_vecs(ce_m, qe_m, chunk_ids, questions)
    _free(h)

    # === 2-5. bge-m3 (общий загрузчик) ===
    print("[2] bge-m3 raw ...")
    gm, h = _make_loader(BGE, device)
    ce_raw, qe_raw, _ = get_embeddings(BGE, False, chunks, questions, gm)
    top5["bge-m3 raw"] = top5_by_vecs(ce_raw, qe_raw, chunk_ids, questions)

    print("[3] bge-m3 + enriched ...")
    ce_enr, qe_enr, _ = get_embeddings(BGE, True, chunks, questions, gm)
    top5["bge-m3 + enriched"] = top5_by_vecs(ce_enr, qe_enr, chunk_ids, questions)

    print("[4] bge-m3 + enriched + HyDE-mix ...")
    hyde_map = hyde.load_or_generate(questions)
    model = gm()
    hyde_emb = np.asarray(model.encode([hyde_map[q["question_id"]] for q in questions],
                                       normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)
    qmix = qe_enr + hyde_emb
    top5["bge-m3 + enriched + HyDE-mix"] = top5_by_vecs(ce_enr, qmix, chunk_ids, questions)

    print("[5] + hybrid RRF (q+hyde, w=0.2) ...")
    bm25 = BM25Okapi([hybrid.tokenize(enriched_map[cid]) for cid in chunk_ids])
    hyb = {}
    for i, q in enumerate(questions):
        d_ids = hybrid.dense_ranking(ce_enr, qmix[i], chunk_ids)
        bm = hybrid.bm25_ranking(bm25, chunk_ids,
                                 hybrid.tokenize(q["query"] + " " + hyde_map[q["question_id"]]))
        hyb[q["question_id"]] = hybrid.rrf(d_ids, bm, w_b=0.2)
    top5["+ HyDE-mix + hybrid (w=0.2)"] = hyb

    # кандидаты для реранкера (enriched-база top-20) — пока есть эмбеддинги bge-m3
    cand_list = [hybrid.dense_ranking(ce_enr, qe_enr[i], chunk_ids)[:rerank.TOP_N]
                 for i in range(len(questions))]
    _free(h)   # bge-m3 больше не нужен

    # === 6. bge-m3 + enriched + rerank ===
    print("[6] bge-m3 + enriched + rerank ...")
    cache_file = CACHE_DIR / f"rerank_scores_{slug(rerank.RERANKER)}.pkl"
    score_cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(rerank.RERANKER, device=device, max_length=512)
    rr = rerank.rerank(ce_model, questions, cand_list, enriched_map, score_cache)
    cache_file.write_bytes(pickle.dumps(score_cache))
    top5["bge-m3 + enriched + rerank"] = {r["question_id"]: r["top_5_chunks"] for r in rr}
    del ce_model
    _free({"m": None})

    # === агрегация + CSV ===
    order = ["baseline (MiniLM raw)", "bge-m3 raw", "bge-m3 + enriched",
             "bge-m3 + enriched + HyDE-mix", "+ HyDE-mix + hybrid (w=0.2)",
             "bge-m3 + enriched + rerank"]
    rows, per_q = [], {}
    for name in order:
        row, sc = aggregate(top5[name], questions)
        per_q[name] = sc
        rows.append({"config": name, **row})

    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en",
            "p5_official15", "p5_extended"]
    csv_path = RESULTS_DIR / "ablation_full.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 118)
    print(f"{'config':<32}{'P@5':>7}{'easy':>7}{'med':>7}{'hard':>7}{'ru':>7}{'en':>7}"
          f"{'offic15':>9}{'ext':>8}")
    print("-" * 118)
    for r in rows:
        print(f"{r['config']:<32}{fmt(r['p5_total']):>7}{fmt(r['p5_easy']):>7}"
              f"{fmt(r['p5_medium']):>7}{fmt(r['p5_hard']):>7}{fmt(r['p5_ru']):>7}"
              f"{fmt(r['p5_en']):>7}{fmt(r['p5_official15']):>9}{fmt(r['p5_extended']):>8}")
    print("=" * 118)

    # === значимость: парный бутстрап разностей ===
    def paired(name_a, name_b):
        a, b = per_q[name_a], per_q[name_b]
        d = np.array([a[q["question_id"]] - b[q["question_id"]] for q in questions])
        n = len(d)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return dict(mean=d.mean(), se=d.std(ddof=1) / np.sqrt(n),
                    ci=(lo, hi), changed=int((d != 0).sum()),
                    up=int((d > 0).sum()), down=int((d < 0).sum()))

    print("\n=== Значимость (парный бутстрап, %d вопросов; 1 вопрос ≈ %.4f) ==="
          % (len(questions), 1 / len(questions)))
    pairs = [
        ("bge-m3 + enriched + HyDE-mix", "bge-m3 + enriched"),          # главная пара
        ("bge-m3 + enriched", "bge-m3 raw"),                            # вклад enrichment
        ("+ HyDE-mix + hybrid (w=0.2)", "bge-m3 + enriched + HyDE-mix"),  # вклад hybrid
        ("bge-m3 + enriched + rerank", "bge-m3 + enriched"),           # вред reranking
    ]
    for a, b in pairs:
        s = paired(a, b)
        sig = "ЗНАЧИМО" if (s["ci"][0] > 0 or s["ci"][1] < 0) else "в пределах шума"
        print(f"\n{a}\n  vs {b}")
        print(f"  Δmean={s['mean']:+.4f}  SE={s['se']:.4f}  95% CI=[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]"
              f"  -> {sig}")
        print(f"  изменилось: {s['changed']} вопросов (лучше {s['up']}, хуже {s['down']})")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

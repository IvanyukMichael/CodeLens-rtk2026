"""Приоритет 3, п.2: код-tuned reranker jina-reranker-v2 на расширенном наборе (170).

Новый «рычаг» ретрива для сравнения в отчёте (рядом с уже отклонённым
bge-reranker-v2-m3 из rerank.py). Пайплайн тот же, что у rerank.py, но реранкер
другой и оценка — на combined-170 (а не на 15, чтобы не оверфитить):

    bge-m3(enriched) -> top-20 кандидатов
        -> cross-encoder jinaai/jina-reranker-v2-base-multilingual (пары запрос/enriched-чанк)
        -> новый top-5

Сравнивает базу (bge-m3 enriched) и базу+jina-rerank на 170 вопросах со срезами
(easy/medium/hard, ru/en, official/extended) и парным бутстрапом значимости.
Пишет results/ablation_rerank_jina.csv. Скоры реранкера кэшируются по парам
(qid, chunk_id) в cache/rerank_scores_jina-reranker-v2-base-multilingual.pkl.

⚠️ ЗАПУСКАТЬ НА МАШИНЕ С CUDA. На CPU 20×170 пар cross-encoder'а непрактично.
jina-reranker-v2 требует trust_remote_code=True (кастомный код модели; может
потребоваться свой набор версий transformers — это на стороне GPU-окружения).

Запуск:
    python exp_rerank_jina.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов HF

import csv
import json
import pickle
from pathlib import Path

import numpy as np

import rerank                                   # rerank(), top_n_candidates(), TOP_N, TOP_K
from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map
from eval_runner import REPO, RESULTS_DIR, CACHE_DIR, get_embeddings, _make_loader, _free, slug
from score import score_question

BGE = "BAAI/bge-m3"
RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
OFFICIAL = Path("eval_questions.json")
EXTENDED = Path("eval/eval_extended.json")
COMBINED = Path("eval/eval_combined.json")


def load_combined() -> list[dict]:
    """Объединённый eval (official + extended) с полем subset. Берём готовый
    eval_combined.json, если он есть; иначе собираем из частей."""
    if COMBINED.exists():
        return json.loads(COMBINED.read_text(encoding="utf-8"))
    official = [{**q, "subset": "official"} for q in json.loads(OFFICIAL.read_text(encoding="utf-8"))]
    extended = [{**q, "subset": "extended"} for q in json.loads(EXTENDED.read_text(encoding="utf-8"))]
    return official + extended


def aggregate(top5_by_qid: dict, questions: list[dict]):
    """Возвращает (row срезов, {qid: score}) — те же срезы, что в ablation_full."""
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
        "p5_official": mean(lambda q: q.get("subset") == "official"),
        "p5_extended": mean(lambda q: q.get("subset") == "extended"),
    }
    return row, sc


def paired_bootstrap(sc_a: dict, sc_b: dict, questions: list[dict], n_boot=10000):
    """Парный бутстрап разностей P@5 (a - b), как в ablation_full."""
    d = np.array([sc_a[q["question_id"]] - sc_b[q["question_id"]] for q in questions])
    n = len(d)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(d.mean()), ci=(float(lo), float(hi)),
                up=int((d > 0).sum()), down=int((d < 0).sum()))


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("⚠️  CUDA не найдена — на CPU этот эксперимент крайне медленный. "
              "Рекомендуется GPU-машина.")

    questions = load_combined()
    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    enriched_map = build_enriched_map(chunks)
    print(f"вопросов: {len(questions)}  |  чанков gymhero: {len(chunks)}")

    # --- база: bge-m3 enriched (эмбеддинги из кэша; модель грузится при промахе) ---
    get_model, holder = _make_loader(BGE, device)
    chunk_emb, query_emb, hit = get_embeddings(BGE, True, chunks, questions, get_model)
    print(f"bge-m3 enriched эмбеддинги: cache_hit={hit}")
    _free(holder)

    # --- кандидаты top-20 + база top-5 ---
    candidates = rerank.top_n_candidates(chunk_emb, query_emb, chunk_ids, rerank.TOP_N)
    base_top5 = {q["question_id"]: candidates[i][:rerank.TOP_K]
                 for i, q in enumerate(questions)}
    print(f"recall@{rerank.TOP_K} (база)   = {rerank.recall_at_n(candidates, questions, rerank.TOP_K):.3f}")
    print(f"recall@{rerank.TOP_N} (потолок) = {rerank.recall_at_n(candidates, questions, rerank.TOP_N):.3f}")

    # --- jina-reranker-v2 ---
    cache_file = CACHE_DIR / f"rerank_scores_{slug(RERANKER)}.pkl"
    score_cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}

    from sentence_transformers import CrossEncoder
    print(f"загрузка реранкера {RERANKER} (trust_remote_code=True) ...")
    ce = CrossEncoder(RERANKER, device=device, max_length=1024, trust_remote_code=True)
    rr = rerank.rerank(ce, questions, candidates, enriched_map, score_cache)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(score_cache))
    rr_top5 = {r["question_id"]: r["top_5_chunks"] for r in rr}

    # --- агрегация + CSV ---
    base_row, base_sc = aggregate(base_top5, questions)
    rr_row, rr_sc = aggregate(rr_top5, questions)
    rows = [{"config": "bge-m3 + enriched", **base_row},
            {"config": "bge-m3 + enriched + jina-rerank-v2", **rr_row}]
    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en",
            "p5_official", "p5_extended"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ablation_rerank_jina.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / "preds_bge-m3_enriched_jina_rerank.json").write_text(
        json.dumps(rr, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 104)
    print(f"{'config':<36}{'P@5':>7}{'easy':>7}{'med':>7}{'hard':>7}{'ru':>7}{'en':>7}"
          f"{'offic':>8}{'ext':>8}")
    print("-" * 104)
    for r in rows:
        print(f"{r['config']:<36}{fmt(r['p5_total']):>7}{fmt(r['p5_easy']):>7}"
              f"{fmt(r['p5_medium']):>7}{fmt(r['p5_hard']):>7}{fmt(r['p5_ru']):>7}"
              f"{fmt(r['p5_en']):>7}{fmt(r['p5_official']):>8}{fmt(r['p5_extended']):>8}")
    print("=" * 104)

    s = paired_bootstrap(rr_sc, base_sc, questions)
    sig = "ЗНАЧИМО" if (s["ci"][0] > 0 or s["ci"][1] < 0) else "в пределах шума"
    print(f"\n=== Значимость jina-rerank vs база (парный бутстрап, n={len(questions)}) ===")
    print(f"  Δmean={s['mean']:+.4f}  95% CI=[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]  -> {sig}")
    print(f"  лучше {s['up']} вопросов / хуже {s['down']}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

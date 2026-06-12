"""Мульти-сэмплинг HyDE поверх лучшей конфигурации (bge-m3 + enriched + HyDE-mix = 0.822).

Идея: одиночная HyDE-генерация шумна. Сгенерим по 3 гипотетических фрагмента на вопрос
(temperature=0.7 + разные seed -> разнообразие), усредним их эмбеддинги (центроид
нормированных векторов, затем ренормировка к единичной длине — чтобы HyDE-компонента
была того же масштаба, что и в single, и сравнение было честным A/B на число сэмплов),
и смешаем с эмбеддингом запроса равновесно (α=0.5), как в одиночном HyDE.

Прогон на полном eval (71), сравнение single vs multi3 со срезами и подмножествами,
парный бутстрап значимости.

Запуск (долгий — 213 генераций; лучше в фоне):
    python hyde_multi.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
from pathlib import Path

import numpy as np
import requests

import hyde
from extract_chunks import extract_repo
from eval_runner import REPO, RESULTS_DIR, get_embeddings, _make_loader, _free
from ablation_full import load_combined, top5_by_vecs, aggregate

BGE = "BAAI/bge-m3"
MULTI_CACHE = Path("cache/hyde_multi.json")
N_SAMPLES = 3
SEEDS = [0, 1, 2]
TEMPERATURE = 0.7


def ollama_gen(question: str, seed: int) -> str:
    r = requests.post(hyde.OLLAMA_URL, json={
        "model": hyde.LLM_MODEL, "system": hyde.SYSTEM_PROMPT, "prompt": question,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 400, "seed": seed},
    }, timeout=180)
    r.raise_for_status()
    return hyde._FENCE.sub("", r.json()["response"].strip()).strip()


def load_or_generate_multi(questions) -> dict:
    cache = json.loads(MULTI_CACHE.read_text(encoding="utf-8")) if MULTI_CACHE.exists() else {}
    todo = [q for q in questions
            if len(cache.get(q["question_id"], [])) < N_SAMPLES]
    if todo:
        print(f"Мульти-генерация ({N_SAMPLES} сэмпла) для {len(todo)} вопросов, temp={TEMPERATURE} ...")
        for i, q in enumerate(todo, 1):
            cache[q["question_id"]] = [ollama_gen(q["query"], s) for s in SEEDS]
            if i % 10 == 0:
                print(f"  {i}/{len(todo)} ...")
                MULTI_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        MULTI_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MULTI_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"Мульти-генерации загружены из кэша: {MULTI_CACHE}")
    return cache


def unit_rows(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    questions = load_combined()
    chunks = extract_repo(REPO)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"вопросов: {len(questions)}")

    single_map = json.loads(hyde.HYDE_CACHE.read_text(encoding="utf-8"))   # 71 одиночных
    multi_map = load_or_generate_multi(questions)

    # bge-m3 enriched чанки/запросы (из кэша) + загрузка модели для эмбеддинга генераций
    gm, h = _make_loader(BGE, device)
    ce_enr, qe_enr, hit = get_embeddings(BGE, True, chunks, questions, gm)
    print(f"bge-m3 enriched эмбеддинги: cache_hit={hit}")
    model = gm()

    # single HyDE -> mix (воспроизводит 0.822)
    h_single = np.asarray(model.encode([single_map[q["question_id"]] for q in questions],
                                        normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)                      # уже единичные
    mix_single = qe_enr + h_single

    # multi HyDE: эмбеддим 71*3, усредняем по 3, ренормируем -> mix
    flat = [g for q in questions for g in multi_map[q["question_id"]][:N_SAMPLES]]
    emb = np.asarray(model.encode(flat, normalize_embeddings=True, show_progress_bar=False),
                     dtype=np.float32).reshape(len(questions), N_SAMPLES, -1)
    h_multi = unit_rows(emb.mean(axis=1))                       # центроид -> единичный
    mix_multi = qe_enr + h_multi
    _free(h)

    top5_single = top5_by_vecs(ce_enr, mix_single, chunk_ids, questions)
    top5_multi = top5_by_vecs(ce_enr, mix_multi, chunk_ids, questions)

    row_s, sc_s = aggregate(top5_single, questions)
    row_m, sc_m = aggregate(top5_multi, questions)
    rows = [{"config": "hyde_mix_single", **row_s},
            {"config": "hyde_mix_multi3", **row_m}]

    cols = ["config", "p5_total", "p5_easy", "p5_medium", "p5_hard", "p5_ru", "p5_en",
            "p5_official15", "p5_extended56"]
    csv_path = RESULTS_DIR / "ablation_hyde_multi.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "

    print("\n" + "=" * 104)
    print(f"{'config':<18}{'P@5':>7}{'easy':>7}{'med':>7}{'hard':>7}{'ru':>7}{'en':>7}"
          f"{'offic15':>9}{'ext56':>8}")
    print("-" * 104)
    for r in rows:
        print(f"{r['config']:<18}{fmt(r['p5_total']):>7}{fmt(r['p5_easy']):>7}{fmt(r['p5_medium']):>7}"
              f"{fmt(r['p5_hard']):>7}{fmt(r['p5_ru']):>7}{fmt(r['p5_en']):>7}"
              f"{fmt(r['p5_official15']):>9}{fmt(r['p5_extended56']):>8}")
    print("=" * 104)

    # парный бутстрап multi3 vs single
    d = np.array([sc_m[q["question_id"]] - sc_s[q["question_id"]] for q in questions])
    n = len(d)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = "ЗНАЧИМО" if (lo > 0 or hi < 0) else "в пределах шума"
    print(f"\n=== multi3 vs single (парный бутстрап, n={n}; 1 вопрос ≈ {1/n:.4f}) ===")
    print(f"  Δmean={d.mean():+.4f}  SE={d.std(ddof=1)/np.sqrt(n):.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {sig}")
    print(f"  изменилось: {int((d!=0).sum())} вопросов (лучше {int((d>0).sum())}, хуже {int((d<0).sum())})")

    # согласованность подмножеств (обобщаемость)
    print("\n=== согласованность official15 vs extended56 ===")
    for r in rows:
        gap = r["p5_official15"] - r["p5_extended56"]
        print(f"  {r['config']:<18} official={fmt(r['p5_official15'])}  "
              f"extended={fmt(r['p5_extended56'])}  |gap|={abs(gap):.3f}")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

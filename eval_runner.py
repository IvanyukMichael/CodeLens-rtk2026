"""Универсальный eval-harness для ablation по моделям/режимам.

Конфигурация = (модель, enriched?). Для каждой:
    чанки -> [enrich?] -> эмбеддинги (кэш на конфиг) -> топ-5 (косинус, numpy)
          -> results/preds_*.json -> score.py -> {P@5, срезы по lang/difficulty}

Запуск:
    python eval_runner.py --all                         # вся матрица 4x2 -> CSV + таблица
    python eval_runner.py --model BAAI/bge-m3 --enriched # одна конфигурация -> dict

Кэш эмбеддингов: cache/{model_slug}_{raw|enriched}_embeddings.pkl
  (хранит и эмбеддинги чанков, и эмбеддинги вопросов — повторный прогон не грузит модель).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import gc
import json
import os
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np

from extract_chunks import extract_repo
from enriched_chunks import build_enriched_map

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO = Path("gymhero")
EVAL = Path("eval_questions.json")
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
]

# Префиксы — обязательны для честного сравнения: e5 требует query:/passage:,
# gte-Qwen2-instruct требует инструкцию у запроса. Иначе модели недооцениваются.
_DEFAULT_CFG = {"query_prefix": "", "doc_prefix": "", "trust_remote_code": False}
MODEL_CONFIG = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": dict(_DEFAULT_CFG),
    "intfloat/multilingual-e5-large": {
        "query_prefix": "query: ", "doc_prefix": "passage: ", "trust_remote_code": False,
    },
    "BAAI/bge-m3": dict(_DEFAULT_CFG),
    # trust_remote_code=False: используем НАТИВНУЮ реализацию Qwen2 из transformers.
    # Родное remote-modeling этой модели обращается к config.rope_theta, которого
    # уже нет в transformers v5 (RoPE-параметры переехали) -> AttributeError.
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "query_prefix": (
            "Instruct: Given a natural-language question about a Python codebase, "
            "retrieve the most relevant code chunk\nQuery: "
        ),
        "doc_prefix": "", "trust_remote_code": False,
    },
}


def slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_name.split("/")[-1]).strip("-")


def cfg_of(model_name: str) -> dict:
    return MODEL_CONFIG.get(model_name, dict(_DEFAULT_CFG))


def cache_path(model_name: str, enriched: bool) -> Path:
    return CACHE_DIR / f"{slug(model_name)}_{'enriched' if enriched else 'raw'}_embeddings.pkl"


def build_docs(chunks: list[dict], enriched: bool) -> list[str]:
    if enriched:
        m = build_enriched_map(chunks)
        return [m[c["chunk_id"]] for c in chunks]
    return [c["code"] for c in chunks]


def get_embeddings(model_name, enriched, chunks, questions, get_model):
    """Возвращает (chunk_emb, query_emb, cache_hit). Нормированные эмбеддинги."""
    cp = cache_path(model_name, enriched)
    chunk_ids = [c["chunk_id"] for c in chunks]
    qids = [q["question_id"] for q in questions]

    if cp.exists():
        with open(cp, "rb") as f:
            cached = pickle.load(f)
        if (cached.get("model") == model_name
                and cached.get("chunk_ids") == chunk_ids
                and cached.get("query_ids") == qids):
            return (np.asarray(cached["chunk_emb"], dtype=np.float32),
                    np.asarray(cached["query_emb"], dtype=np.float32), True)

    cfg = cfg_of(model_name)
    model = get_model()
    docs = [cfg["doc_prefix"] + d for d in build_docs(chunks, enriched)]
    queries = [cfg["query_prefix"] + q["query"] for q in questions]

    chunk_emb = np.asarray(
        model.encode(docs, normalize_embeddings=True, batch_size=16,
                     show_progress_bar=False), dtype=np.float32)
    query_emb = np.asarray(
        model.encode(queries, normalize_embeddings=True, batch_size=16,
                     show_progress_bar=False), dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump({"model": model_name, "chunk_ids": chunk_ids, "query_ids": qids,
                     "chunk_emb": chunk_emb, "query_emb": query_emb}, f)
    return chunk_emb, query_emb, False


def retrieve_top5(chunk_emb, query_emb, chunk_ids, questions, k=5):
    results = []
    for i, q in enumerate(questions):
        sims = chunk_emb @ query_emb[i]               # косинус (всё нормировано)
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        results.append({"question_id": q["question_id"],
                        "top_5_chunks": [chunk_ids[j] for j in idx]})
    return results


def run_score(pred_path: Path) -> dict:
    """Запускает score.py и парсит Total + срезы из его stdout."""
    proc = subprocess.run(
        [sys.executable, "score.py", "--predictions", str(pred_path),
         "--questions", str(EVAL)],
        capture_output=True, text=True, encoding="utf-8",
    )
    out = proc.stdout

    def grab(pat):
        m = re.search(pat, out, re.MULTILINE)
        return float(m.group(1)) if m else float("nan")

    return {
        "p5_total": grab(r"Total score:\s+([0-9.]+)"),
        "p5_easy": grab(r"^\s+easy\s+([0-9.]+)"),
        "p5_medium": grab(r"^\s+medium\s+([0-9.]+)"),
        "p5_hard": grab(r"^\s+hard\s+([0-9.]+)"),
        "p5_ru": grab(r"^\s+ru:\s+([0-9.]+)"),
        "p5_en": grab(r"^\s+en:\s+([0-9.]+)"),
    }


def run_config(model_name, enriched, get_model, chunks, questions) -> dict:
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_emb, query_emb, hit = get_embeddings(
        model_name, enriched, chunks, questions, get_model)
    results = retrieve_top5(chunk_emb, query_emb, chunk_ids, questions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"preds_{slug(model_name)}_{'enriched' if enriched else 'raw'}.json"
    pred_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    row = run_score(pred_path)
    row.update({"model": model_name, "enriched": enriched, "cache_hit": hit})
    return row


def _free(model_holder):
    if model_holder["m"] is not None:
        del model_holder["m"]
        model_holder["m"] = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _make_loader(model_name, device):
    cfg = cfg_of(model_name)
    holder = {"m": None}

    def get_model():
        if holder["m"] is None:
            from sentence_transformers import SentenceTransformer
            print(f"      загрузка модели (trust_remote_code={cfg['trust_remote_code']}) ...")
            holder["m"] = SentenceTransformer(
                model_name, device=device, trust_remote_code=cfg["trust_remote_code"])
        return holder["m"]

    return get_model, holder


CSV_COLS = ["model", "enriched", "p5_total", "p5_easy", "p5_medium",
            "p5_hard", "p5_ru", "p5_en"]


def write_csv(rows, path=RESULTS_DIR / "ablation_models.csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in CSV_COLS})
    print(f"\nCSV: {path}")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) and x == x else " nan "


def print_table(rows):
    print("\n" + "=" * 92)
    print(f"{'model':<30} {'mode':<9} {'P@5':>6} {'easy':>6} {'med':>6} "
          f"{'hard':>6} {'ru':>6} {'en':>6}")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: (-(r["p5_total"] if r["p5_total"] == r["p5_total"] else -1))):
        name = r["model"].split("/")[-1]
        mode = "enriched" if r["enriched"] else "raw"
        print(f"{name:<30} {mode:<9} {_fmt(r['p5_total']):>6} {_fmt(r['p5_easy']):>6} "
              f"{_fmt(r['p5_medium']):>6} {_fmt(r['p5_hard']):>6} "
              f"{_fmt(r['p5_ru']):>6} {_fmt(r['p5_en']):>6}")
    print("=" * 92)


def run_all():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    chunks = extract_repo(REPO)
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    print(f"чанков: {len(chunks)}, вопросов: {len(questions)}\n")

    rows = []
    for model_name in MODELS:
        print(f"\n########## {model_name} ##########")
        get_model, holder = _make_loader(model_name, device)
        for enriched in (False, True):
            mode = "enriched" if enriched else "raw"
            print(f"  -> {mode}")
            try:
                row = run_config(model_name, enriched, get_model, chunks, questions)
                print(f"     P@5={_fmt(row['p5_total'])} "
                      f"(ru={_fmt(row['p5_ru'])} en={_fmt(row['p5_en'])} "
                      f"cache_hit={row['cache_hit']})")
            except Exception as e:
                print(f"     ОШИБКА: {type(e).__name__}: {e}")
                row = {"model": model_name, "enriched": enriched,
                       "p5_total": float("nan"), "p5_easy": float("nan"),
                       "p5_medium": float("nan"), "p5_hard": float("nan"),
                       "p5_ru": float("nan"), "p5_en": float("nan")}
            rows.append(row)
        _free(holder)

    write_csv(rows)
    print_table(rows)
    return rows


def run_single(model_name, enriched):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = extract_repo(REPO)
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    get_model, holder = _make_loader(model_name, device)
    try:
        row = run_config(model_name, enriched, get_model, chunks, questions)
    finally:
        _free(holder)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CodeLens ablation eval-harness")
    ap.add_argument("--model", help="HF id модели для одиночного прогона")
    ap.add_argument("--enriched", action="store_true", help="режим enrichment")
    ap.add_argument("--all", action="store_true", help="вся матрица 4 модели x {raw,enriched}")
    args = ap.parse_args()

    if args.all or not args.model:
        run_all()
    else:
        run_single(args.model, args.enriched)

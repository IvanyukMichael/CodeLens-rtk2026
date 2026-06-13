"""CodeLens — сборка официального results.json финальной конфигурацией.

Прогоняет ЗАМОРОЖЕННОЕ ядро поиска [search.py] (bge-m3 + enriched + HyDE-mix,
temp=0) по официальным вопросам eval_questions.json и пишет верхнеуровневый
results.json в формате сдачи организаторов:

    [ { "question_id": ..., "top_5_chunks": [<5 chunk_id>] }, ... ]

Это и есть файл-сдача. ВНИМАНИЕ: baseline.py тоже пишет results.json (как
побочный эффект своего прогона), поэтому для ОФИЦИАЛЬНОЙ выдачи всегда
перегенерируй файл этим скриптом — он использует финальную конфигурацию.

Детерминированно: HyDE temp=0 (greedy). Если все запросы есть в
cache/hyde_query_cache.json — Ollama не нужна; иначе генерация идёт живьём
(тоже temp=0). Если Ollama недоступна И есть промах кэша, скрипт НЕ перезаписывает
results.json (чтобы не положить туда выдачу без HyDE), а падает с понятным сообщением.

Запуск:
    python predict.py                                   # -> results.json
    python predict.py --eval eval_questions.json --out results.json
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
from pathlib import Path

import search   # финальное ядро (внутри импортируется sslfix до сетевых вызовов HF)


def build_predictions(questions: list[dict]) -> list[dict]:
    """top-5 финальной конфигурацией на каждый вопрос. Бросает, если HyDE
    откатился (нет Ollama и промах кэша) — чтобы не записать выдачу без HyDE."""
    search.warmup()                                     # bge-m3 + ChromaDB один раз
    preds, fell_back = [], []
    for q in questions:
        r = search.search(q["query"], top_k=5, use_hyde=True)
        if not r["hyde_used"]:
            fell_back.append((q["question_id"], r.get("warning")))
        top5 = [h["chunk_id"] for h in r["results"]]
        if len(top5) != 5:
            raise SystemExit(f"{q['question_id']}: получено {len(top5)} чанков (ожидалось 5)")
        preds.append({"question_id": q["question_id"], "top_5_chunks": top5})

    if fell_back:
        msg = ["ОТМЕНА — HyDE откатился (нет Ollama и промах кэша); results.json НЕ изменён:"]
        msg += [f"  {qid}: {w}" for qid, w in fell_back]
        raise SystemExit("\n".join(msg))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="CodeLens — сборка официального results.json")
    ap.add_argument("--eval", default="eval_questions.json", help="файл вопросов")
    ap.add_argument("--out", default="results.json", help="куда писать предсказания")
    args = ap.parse_args()

    questions = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    preds = build_predictions(questions)

    Path(args.out).write_text(
        json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.out}: {len(preds)} вопросов, все с HyDE-mix (temp=0).")
    print("top-1 на вопрос:")
    for p in preds:
        print(f"  {p['question_id']}: {p['top_5_chunks'][0]}")


if __name__ == "__main__":
    main()

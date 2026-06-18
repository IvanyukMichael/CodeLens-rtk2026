"""Расширение eval до ~170 вопросов через LLM-автора (Claude), без Ollama.

Зачем: gen_eval.py генерирует вопросы локальной Ollama (qwen2.5-coder:7b). Когда
Ollama недоступна, автором выступает Claude — по ТЕМ ЖЕ промтам/правилам и с теми же
детекторами качества из gen_eval.py (утечка имени, копирование few-shot). Это
методологический эквивалент (сильнее по модели), результат честно помечается как
LLM-generated + detector-validated.

Конвейер из двух шагов:
  1) `python gen_eval_llm.py worktable`
     Отбирает чанки-кандидаты gymhero (как gen_eval), планирует пары (chunk, lang)
     так, чтобы суммарно (с уже существующими 56) получилось ~155 расширенных, и
     пишет eval/_worktable.json с кодом каждого чанка — это «задание» автору.
  2) автор (Claude) заполняет eval/_authored.json: {question_id: "текст вопроса"}.
  3) `python gen_eval_llm.py assemble`
     Валидирует каждый вопрос детекторами gen_eval, собирает eval/eval_extended.json
     (старые 56 + новые) и пересобирает eval/eval_combined.json (official+extended).

Существующие 56 (g_01..g_56) сохраняются без изменений.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
from collections import Counter
from pathlib import Path

from extract_chunks import extract_repo
import gen_eval as G

REPO = Path("gymhero")
ORIG_EVAL = Path("eval_questions.json")
EXTENDED = Path("eval/eval_extended.json")
COMBINED = Path("eval/eval_combined.json")
WORKTABLE = Path("eval/_worktable.json")
AUTHORED = Path("eval/_authored.json")

TARGET_EXTENDED = 155          # цель: 15 official + 155 extended = 170 combined


def _candidates():
    chunks = extract_repo(REPO)
    orig = json.loads(ORIG_EVAL.read_text(encoding="utf-8"))
    excluded = {cid for q in orig for cid in q["correct_chunk_ids"]}
    cands = [c for c in chunks if G.is_candidate(c, excluded)]
    return {c["chunk_id"]: c for c in cands}


def build_worktable():
    cand = _candidates()
    existing = json.loads(EXTENDED.read_text(encoding="utf-8"))
    # (chunk_id, lang) уже занятые существующими вопросами
    taken = {(q["correct_chunk_ids"][0], q["language"]) for q in existing}
    used_ids = {q["question_id"] for q in existing}
    lang_balance = Counter(q["language"] for q in existing)

    def next_id(n=[max((int(i.split("_")[1]) for i in used_ids), default=0)]):
        n[0] += 1
        return f"g_{n[0]:02d}"

    def pick_lang():
        # выравниваем общий баланс ru/en
        return "ru" if lang_balance["ru"] <= lang_balance["en"] else "en"

    targets = []

    # Фаза A: каждый кандидат, ещё не покрытый НИ одним языком, получает 1 вопрос.
    covered = {cid for (cid, _l) in taken}
    new_chunks = [cid for cid in cand if cid not in covered]
    # порядок детерминирован: по difficulty (hard->easy для приоритета сложных) + path
    new_chunks.sort(key=lambda cid: (
        {"hard": 0, "medium": 1, "easy": 2}[G.difficulty_of(cand[cid])], cand[cid]["path"]))
    for cid in new_chunks:
        if len(existing) + len(targets) >= TARGET_EXTENDED:
            break
        lang = pick_lang()
        lang_balance[lang] += 1
        targets.append((cid, lang))

    # Фаза B: добиваем до цели вторым языком у уже покрытых чанков (cross-language).
    if len(existing) + len(targets) < TARGET_EXTENDED:
        planned = {(cid, lang) for cid, lang in targets}
        # кандидаты на 2-й язык: чанк покрыт одним языком, другой ещё не занят
        second = []
        have_lang = {}
        for (cid, l) in taken:
            have_lang.setdefault(cid, set()).add(l)
        for cid, lang in targets:
            have_lang.setdefault(cid, set()).add(lang)
        for cid in sorted(cand, key=lambda c: (
                {"hard": 0, "medium": 1, "easy": 2}[G.difficulty_of(cand[c])], cand[c]["path"])):
            have = have_lang.get(cid, set())
            for lang in ("en", "ru"):
                if lang not in have and (cid, lang) not in taken and (cid, lang) not in planned:
                    second.append((cid, lang))
        for cid, lang in second:
            if len(existing) + len(targets) >= TARGET_EXTENDED:
                break
            lang_balance[lang] += 1
            targets.append((cid, lang))

    rows = []
    for cid, lang in targets:
        c = cand[cid]
        rows.append({
            "question_id": next_id(),
            "chunk_id": cid, "language": lang,
            "path": c["path"], "name": c["name"], "kind": c["kind"],
            "difficulty": G.difficulty_of(c), "category": G.category_of(c),
            "is_model": G.is_data_model(c),
            "code": c["code"],
        })

    WORKTABLE.parent.mkdir(parents=True, exist_ok=True)
    WORKTABLE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # пустой шаблон для авторских вопросов (если ещё нет)
    if not AUTHORED.exists():
        AUTHORED.write_text(json.dumps({r["question_id"]: "" for r in rows},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"worktable: {len(rows)} новых заданий -> {WORKTABLE}")
    print(f"итоговый extended будет: {len(json.loads(EXTENDED.read_text(encoding='utf-8'))) + len(rows)}")
    print("по языку (новые):    ", dict(Counter(r["language"] for r in rows)))
    print("по сложности (новые):", dict(Counter(r["difficulty"] for r in rows)))
    print("по категории (новые):", dict(Counter(r["category"] for r in rows)))


def assemble():
    rows = json.loads(WORKTABLE.read_text(encoding="utf-8"))
    authored = json.loads(AUTHORED.read_text(encoding="utf-8"))
    cand = _candidates()
    existing = json.loads(EXTENDED.read_text(encoding="utf-8"))

    problems, new_records = [], []
    for r in rows:
        qid = r["question_id"]
        q = (authored.get(qid) or "").strip()
        if not q:
            problems.append(f"{qid}: пустой вопрос"); continue
        chunk = cand[r["chunk_id"]]
        kind = "model" if r["is_model"] else "behavior"
        examples = G.EX[(r["language"], kind)]
        if G.leaks_name(q, chunk):
            problems.append(f"{qid}: утечка имени -> {q!r}"); continue
        if G.too_similar(q, examples):
            problems.append(f"{qid}: слишком похоже на few-shot -> {q!r}"); continue
        new_records.append({
            "question_id": qid, "query": q, "language": r["language"],
            "correct_chunk_ids": [r["chunk_id"]],
            "difficulty": r["difficulty"], "category": r["category"],
        })

    if problems:
        print("ПРОБЛЕМЫ (исправь eval/_authored.json и повтори assemble):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    full = existing + new_records
    EXTENDED.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

    # пересобрать combined: official(subset=official) + extended(subset=extended)
    official = json.loads(ORIG_EVAL.read_text(encoding="utf-8"))
    combined = ([{**q, "subset": "official"} for q in official]
                + [{**q, "subset": "extended"} for q in full])
    COMBINED.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"extended: {len(full)} вопросов -> {EXTENDED}")
    print(f"combined: {len(combined)} вопросов -> {COMBINED}")
    print("extended по языку:    ", dict(Counter(r["language"] for r in full)))
    print("extended по сложности:", dict(Counter(r["difficulty"] for r in full)))
    print("extended по категории:", dict(Counter(r["category"] for r in full)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worktable", "assemble"])
    args = ap.parse_args()
    if args.cmd == "worktable":
        build_worktable()
    else:
        assemble()


if __name__ == "__main__":
    main()

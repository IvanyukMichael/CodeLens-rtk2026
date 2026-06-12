"""Генератор расширенного eval (§7 CLAUDE.md).

Берём чанки gymhero (НЕ пересекающиеся с эталонными 15 вопросами), для каждого
просим Ollama (qwen2.5-coder:7b) сформулировать РЕАЛИСТИЧНЫЙ поисковый вопрос —
как от разработчика, который ищет, где что реализовано (а НЕ «что делает функция X»).
Отдельные промты на русский/английский И отдельные — для функций vs моделей данных
(Pydantic-схемы / ORM-модели), т.к. у DTO нет «поведения» и общий промт даёт мусор.

Защиты от типичных дефектов:
- утечка имени функции/класса в вопрос -> ретрай;
- копирование few-shot примера дословно (близкий дубликат) -> ретрай.

Формат записи — как в eval_questions.json:
    {question_id, query, language, correct_chunk_ids, difficulty, category}
correct_chunk_ids = [chunk_id исходной функции] (связанные чанки для hard — вручную позже).

Результат -> eval/eval_extended.json (оригинальный eval_questions.json НЕ трогаем).
Генерации кэшируются в cache/eval_gen_cache.json (ключ включает версию промта).

Запуск:
    python gen_eval.py            # сгенерировать (или дочитать из кэша) + сводка + 10 примеров
    python gen_eval.py --n 56
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import requests

from extract_chunks import extract_repo

REPO = Path("gymhero")
ORIG_EVAL = Path("eval_questions.json")
OUT = Path("eval/eval_extended.json")
GEN_CACHE = Path("cache/eval_gen_cache.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5-coder:7b"
PROMPT_VERSION = "v2"

AUTH_KW = ("token", "password", "jwt", "superuser", "authenticate",
           "login", "credential", "oauth")

# ---------- few-shot примеры (стиль) ----------
EX = {
    ("ru", "behavior"): [
        "как в проекте создаётся токен доступа и какой срок его жизни?",
        "где проверяется, является ли пользователь суперпользователем?",
        "как формируется строка подключения к базе данных из настроек?",
    ],
    ("en", "behavior"): [
        "how does the project verify a JWT token from an incoming request?",
        "how does the project create a new account and what response is sent if the email is taken?",
        "how are database pagination parameters validated?",
    ],
    ("ru", "model"): [
        "где описывается структура ответа с данными пользователя?",
        "какие поля задаются при создании нового упражнения?",
        "где определена модель таблицы для хранения планов тренировок?",
    ],
    ("en", "model"): [
        "where is the data structure for a user response defined?",
        "what fields are required to create a new exercise?",
        "where is the database table model for training plans defined?",
    ],
}

_RU_BEHAVIOR = (
    "Ты помогаешь собрать датасет для оценки семантического поиска по коду. Тебе "
    "дают фрагмент Python-кода (функция) из реального backend-проекта. Сформулируй "
    "реалистичный вопрос НА РУССКОМ, на который этот фрагмент — правильный ответ.\n"
    "Требования:\n"
    "- Поисковый запрос: «где/как в проекте ...», про поведение и назначение кода.\n"
    "- НЕ упоминай имя функции/класса и точные идентификаторы из кода.\n"
    "- НЕ пиши «что делает функция X».\n"
    "- НЕ копируй примеры дословно — они только для стиля.\n"
    "- Ровно ОДИН вопрос, одной строкой, без кавычек, нумерации и пояснений."
)
_EN_BEHAVIOR = (
    "You help build a dataset for evaluating semantic code search. You are given a "
    "Python snippet (a function) from a real backend project. Write a realistic "
    "question IN ENGLISH that this snippet correctly answers.\n"
    "Requirements:\n"
    "- A search query: 'how/where does the project ...', about behavior/purpose.\n"
    "- Do NOT mention the function/class name or exact identifiers.\n"
    "- Do NOT write 'what does function X do'.\n"
    "- Do NOT copy the examples verbatim — they only show style.\n"
    "- Exactly ONE question, single line, no quotes, no numbering, no explanation."
)
_RU_MODEL = (
    "Ты помогаешь собрать датасет для оценки поиска по коду. Тебе дают Python-класс — "
    "модель данных (Pydantic-схема или ORM-модель таблицы) из backend-проекта. "
    "Сформулируй реалистичный вопрос НА РУССКОМ, на который этот класс — правильный "
    "ответ: разработчик ищет, ГДЕ описана структура данных / какие поля у сущности / "
    "где задаётся схема ответа или таблицы.\n"
    "Требования:\n"
    "- Поисковый запрос: «где описывается структура ...», «какие поля у ...», "
    "«где задаётся схема/модель ...».\n"
    "- НЕ упоминай имя класса дословно.\n"
    "- НЕ копируй примеры дословно — они только для стиля.\n"
    "- Ровно ОДИН вопрос, одной строкой, без кавычек и пояснений."
)
_EN_MODEL = (
    "You help build a dataset for evaluating code search. You are given a Python class "
    "— a data model (Pydantic schema or ORM table model) from a backend project. Write "
    "a realistic question IN ENGLISH this class answers: a developer searching WHERE a "
    "data structure is defined / what fields an entity has / where a response or table "
    "schema is declared.\n"
    "Requirements:\n"
    "- A search query: 'where is the ... data structure defined', 'what fields does ... "
    "have', 'where is the schema/model for ...'.\n"
    "- Do NOT mention the class name verbatim.\n"
    "- Do NOT copy the examples verbatim — they only show style.\n"
    "- Exactly ONE question, single line, no quotes, no explanation."
)
SYSTEMS = {
    ("ru", "behavior"): _RU_BEHAVIOR, ("en", "behavior"): _EN_BEHAVIOR,
    ("ru", "model"): _RU_MODEL, ("en", "model"): _EN_MODEL,
}
NO_COPY_HINT = {
    "ru": "\nВАЖНО: не копируй примеры и не упоминай имя; задай вопрос конкретно про ЭТОТ код.",
    "en": "\nIMPORTANT: do not copy the examples and do not mention the name; ask specifically about THIS code.",
}


# ---------- классификация чанков ----------

def category_of(chunk: dict) -> str:
    path = chunk["path"].lower()
    name = chunk["name"].lower()
    if any(k in name for k in AUTH_KW) or "security.py" in path or "/auth" in path:
        return "auth"
    if "config.py" in path:
        return "config"
    if "/schemas/" in path:
        return "validation"
    if "/api/" in path:
        return "api"
    if any(s in path for s in ("/database/", "/crud/", "/models/", "session", "db.py")):
        return "db"
    return "business_logic"


def is_data_model(chunk: dict) -> bool:
    return chunk["kind"] == "class" and (
        "/schemas/" in chunk["path"] or "/models/" in chunk["path"])


def difficulty_of(chunk: dict) -> str:
    n_lines = chunk["end"] - chunk["start"] + 1
    pts = 0
    if not chunk["docstring"]:
        pts += 1
    if n_lines >= 22:
        pts += 1
    if chunk["kind"] == "function" and "." in chunk["name"]:   # метод внутри класса
        pts += 1
    return ("easy", "medium", "hard", "hard")[min(pts, 3)]


def is_candidate(chunk: dict, excluded: set) -> bool:
    if chunk["chunk_id"] in excluded:
        return False
    if chunk["end"] - chunk["start"] + 1 < 3:
        return False
    if chunk["name"].split(".")[-1] in {"__repr__", "__str__"}:
        return False
    return True


def select_chunks(chunks, excluded, n_target, seed=42):
    """Стратифицированный round-robin: цикл по сложности (для спреда easy/medium/hard),
    внутри — ротация по категориям (для покрытия). Язык назначаем 50/50."""
    rng = random.Random(seed)
    cands = [c for c in chunks if is_candidate(c, excluded)]
    buck = {d: defaultdict(list) for d in ("easy", "medium", "hard")}
    for c in cands:
        buck[difficulty_of(c)][category_of(c)].append(c)
    for d in buck:
        for cat in buck[d]:
            rng.shuffle(buck[d][cat])

    diffs = ["easy", "medium", "hard"]
    cursor = {d: 0 for d in diffs}
    selected, progress = [], True
    while len(selected) < n_target and progress:
        progress = False
        for d in diffs:
            cats = [c for c in sorted(buck[d]) if buck[d][c]]
            if not cats:
                continue
            cat = cats[cursor[d] % len(cats)]
            cursor[d] += 1
            selected.append(buck[d][cat].pop())
            progress = True
            if len(selected) >= n_target:
                break

    rng.shuffle(selected)
    return [(c, "ru" if i % 2 == 0 else "en") for i, c in enumerate(selected)]


# ---------- генерация ----------

def code_view(chunk: dict) -> str:
    body = chunk["code"]
    if len(body) > 1500:
        body = body[:1500] + "\n# ... (truncated)"
    return f"# file: {chunk['path']}\n{body}"


def ollama(system: str, prompt: str) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL, "system": system, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.5, "num_predict": 120},
    }, timeout=180)
    r.raise_for_status()
    return r.json()["response"]


def clean_question(text: str) -> str:
    t = text.strip()
    for line in t.splitlines():
        if line.strip():
            t = line.strip()
            break
    t = t.strip().strip('"\'«»“”')
    t = re.sub(r"^(вопрос|question)\s*[:\-]\s*", "", t, flags=re.I)
    t = re.sub(r"^\d+[.)]\s*", "", t)
    return t.strip()


def _words(s: str) -> set:
    return set(re.findall(r"\w+", s.lower()))


def too_similar(q: str, examples) -> bool:
    """Близкий дубликат few-shot примера (Jaccard по словам)."""
    qw = _words(q)
    if not qw:
        return True
    return any(len(qw & _words(e)) / len(qw | _words(e)) >= 0.6 for e in examples)


def leaks_name(question: str, chunk: dict) -> bool:
    """Утечка идентификатора: любая многословная часть имени (snake_case ИЛИ
    CamelCase, длиной >=6) встретилась в вопросе. Одиночные общие слова
    (get/login/Settings/Token) не считаем утечкой."""
    q = question.lower()
    for part in chunk["name"].split("."):
        multiword = ("_" in part) or bool(re.search(r"[a-z][A-Z]", part))
        if multiword and len(part) >= 6 and part.lower() in q:
            return True
    return False


def gen_question(chunk, lang, cache) -> str:
    key = f"{chunk['chunk_id']}::{lang}::{PROMPT_VERSION}"
    kind = "model" if is_data_model(chunk) else "behavior"
    system = SYSTEMS[(lang, kind)]
    examples = EX[(lang, kind)]
    code = code_view(chunk)

    cached = cache.get(key)        # перепроверяем кэш текущими детекторами (само-починка)
    if cached and not leaks_name(cached, chunk) and not too_similar(cached, examples):
        return cached

    q = clean_question(ollama(system, code))
    if leaks_name(q, chunk) or too_similar(q, examples):       # ретрай при утечке/копировании
        q2 = clean_question(ollama(system + NO_COPY_HINT[lang], code))
        if q2 and not leaks_name(q2, chunk) and not too_similar(q2, examples):
            q = q2
    cache[key] = q
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=56)
    args = ap.parse_args()

    chunks = extract_repo(REPO)
    orig = json.loads(ORIG_EVAL.read_text(encoding="utf-8"))
    excluded = {cid for q in orig for cid in q["correct_chunk_ids"]}
    print(f"чанков всего: {len(chunks)}, исключено (из эталонных 15): {len(excluded)}")

    picked = select_chunks(chunks, excluded, args.n)
    print(f"отобрано чанков: {len(picked)}")

    cache = json.loads(GEN_CACHE.read_text(encoding="utf-8")) if GEN_CACHE.exists() else {}
    n_before = len(cache)

    records = []
    for i, (chunk, lang) in enumerate(picked, 1):
        query = gen_question(chunk, lang, cache)
        records.append({
            "question_id": f"g_{i:02d}",
            "query": query,
            "language": lang,
            "correct_chunk_ids": [chunk["chunk_id"]],
            "difficulty": difficulty_of(chunk),
            "category": category_of(chunk),
        })
        if len(cache) != n_before and i % 10 == 0:
            print(f"  сгенерировано {i}/{len(picked)} ...")

    GEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GEN_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nЗаписано: {OUT}  ({len(records)} вопросов; новых генераций: {len(cache) - n_before})")
    print("по языку:     ", dict(Counter(r["language"] for r in records)))
    print("по сложности: ", dict(Counter(r["difficulty"] for r in records)))
    print("по категории: ", dict(Counter(r["category"] for r in records)))

    print("\n=== 10 случайных вопросов для ручной проверки ===")
    for r in random.Random(7).sample(records, min(10, len(records))):
        print(f"\n[{r['question_id']}] ({r['language']}, {r['difficulty']}, {r['category']})")
        print(f"  Q: {r['query']}")
        print(f"  -> {r['correct_chunk_ids'][0]}")


if __name__ == "__main__":
    main()

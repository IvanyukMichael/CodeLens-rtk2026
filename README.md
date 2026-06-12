# Датасет для Кейса 3 «CodeLens RAG»

> Чемпионат ГК «Ростелеком», 2-й этап, направление «Искусственный интеллект»
> Версия датасета: 1.0 | Дата: Май 2026

---

## Описание датасета

Этот пакет содержит все необходимые данные для решения Кейса 3 «CodeLens RAG». Задача команды — построить систему семантического поиска по Python-кодовой базе: принять запрос на естественном языке, проиндексировать код с помощью векторных эмбеддингов и вернуть топ-5 наиболее релевантных фрагментов. Точность измеряется метрикой Precision@5 — скрипт `score.py` считает её автоматически по эталонным вопросам из `eval_questions.json`. Оценка per-question нормализована формулой `matched / min(5, len(correct))`, что гарантирует 1.0 за идеальный ответ независимо от числа эталонов в вопросе.

Кодовая база — open-source FastAPI-приложение `gymhero` (управление тренировочными планами), реализованное на Python 3.10+ с PostgreSQL и SQLAlchemy. Лицензия MIT.

---

## Содержимое пакета

### `codebase_python.zip`

Зафиксированный снимок исходного кода репозитория [gymhero](https://github.com/JakubPluta/gymhero). Архив очищен: убраны тесты, миграции, lock-файлы, кеш. При распаковке в текущую директорию создаётся папка `gymhero/` с вложенной структурой модулей.

Структура (после распаковки):

```
gymhero/
├── gymhero/
│   ├── api/
│   │   ├── dependencies.py      # FastAPI-зависимости: auth, pagination
│   │   └── routes/              # endpoint-файлы по сущностям
│   ├── crud/
│   │   ├── base.py              # базовый CRUD-репозиторий (CRUDRepository)
│   │   ├── user.py, training_plan.py, training_unit.py, ...
│   ├── database/
│   │   ├── db.py                # сессия БД для FastAPI
│   │   └── session.py           # построение URL, создание engine
│   ├── models/                  # SQLAlchemy ORM-модели
│   ├── schemas/                 # Pydantic-схемы (in/out)
│   ├── config.py                # настройки через pydantic-settings
│   ├── exceptions.py            # кастомные исключения
│   ├── security.py              # JWT, хеширование паролей
│   └── main.py                  # приложение FastAPI
├── pyproject.toml
└── README.md
```

**Что индексировать:** все `.py` файлы в папке `gymhero/gymhero/`.

### `eval_questions.json`

Финальный тестовый набор из 15 вопросов. Каждый вопрос имеет поле `correct_chunk_ids` — список идентификаторов фрагментов, которые ваша система должна поднять в топ-5. Именно по этому файлу считается итоговая метрика.

Формат вопроса:

```json
{
  "question_id": "q_01",
  "query": "как в проекте создаётся токен доступа и какой срок его жизни?",
  "language": "ru",
  "correct_chunk_ids": [
    "gymhero/security.py:create_access_token:12",
    "gymhero/config.py:Settings:11",
    "gymhero/api/routes/auth.py:login_for_access_token:19"
  ],
  "difficulty": "easy",
  "category": "auth"
}
```

Распределение:
- Языки: 8 вопросов на русском, 7 на английском
- Сложность: 5 easy, 6 medium, 4 hard
- Категории: auth, api, db, validation, config, errors, business_logic
- Число эталонов: 4 вопроса с 1 эталоном, 7 с 2 эталонами, 4 с 3 эталонами

### `sample_queries.txt`

20 примеров запросов для демонстрации возможностей системы. Это не тестовый набор — никаких эталонных ответов нет. Используйте их для презентации и отладки в процессе разработки. В файле намеренно есть 3 запроса о функциональности, которой в `gymhero` нет (blockchain, WebSocket, rate limiting) — хорошая RAG-система должна корректно возвращать нерелевантные результаты на такие запросы.

### `score.py`

Автоматический скорер. Принимает ваш файл предсказаний и считает Precision@5. Работает на стандартной библиотеке Python 3.10+, без дополнительных зависимостей.

---

## Спецификация формата `chunk_id`

Все идентификаторы фрагментов — как в `correct_chunk_ids`, так и в вашем `results.json` — должны строго следовать единому формату:

```
{relative_path}:{name}:{start_line}
```

**Поля:**

| Поле | Описание | Пример |
|------|----------|--------|
| `relative_path` | Путь от корня репозитория, прямые слэши `/` | `gymhero/api/dependencies.py` |
| `name` | Имя функции или класса; для метода класса — `ClassName.method_name` | `CRUDRepository.get_many` |
| `start_line` | Номер первой строки определения (`def` или `class`) как возвращает `ast.parse` Python | `53` |

**Полные примеры:**

```
gymhero/security.py:create_access_token:12
gymhero/crud/base.py:CRUDRepository.get_many:53
gymhero/crud/base.py:CRUDRepository.create_with_owner:162
gymhero/api/routes/auth.py:login_for_access_token:19
gymhero/models/training_plan.py:TrainingPlan:23
```

**ВАЖНО:** При сравнении `start_line` допускается отклонение **±2 строки**. Например, если эталон `gymhero/security.py:create_access_token:12`, то ваш результат `gymhero/security.py:create_access_token:10`, `11`, `12`, `13` или `14` — все засчитаются. Это реализовано в `score.py` автоматически.

**Как извлечь chunk_id с помощью `ast`:**

```python
import ast
from pathlib import Path

def extract_chunks(py_file: Path, repo_root: Path):
    """Extract chunk_ids from a Python file using AST."""
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            chunk_id = f"{rel}:{node.name}:{node.lineno}"
            chunks.append(chunk_id)
            # Methods inside the class
            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{rel}:{node.name}.{item.name}:{item.lineno}"
                    chunks.append(method_id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions (check not inside a class)
            chunk_id = f"{rel}:{node.name}:{node.lineno}"
            chunks.append(chunk_id)
    
    return chunks
```

---

## Минимальный пример workflow

### 1. Распаковать кодовую базу

```python
import zipfile
from pathlib import Path

with zipfile.ZipFile("codebase_python.zip", "r") as z:
    z.extractall(".")
# Теперь доступна папка gymhero/
```

### 2. Проиндексировать все Python-файлы

```python
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer  # или любая другая модель

model = SentenceTransformer("BAAI/bge-m3")  # пример — выберите модель самостоятельно
repo_root = Path("gymhero")
index = {}  # chunk_id -> embedding

for py_file in repo_root.rglob("*.py"):
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    
    lines = src.splitlines()
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Определите имя chunk
            name = node.name
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(lines[start:end])
            chunk_id = f"{rel}:{name}:{node.lineno}"
            
            embedding = model.encode(chunk_text)
            index[chunk_id] = embedding

print(f"Indexed {len(index)} chunks")
```

### 3. Прогнать вопросы из eval_questions.json

```python
import json
import numpy as np

questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
results = []

for q in questions:
    query_embedding = model.encode(q["query"])
    
    # Найти топ-5 по косинусному сходству
    scores = {}
    for chunk_id, emb in index.items():
        similarity = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-9
        )
        scores[chunk_id] = similarity
    
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    results.append({"question_id": q["question_id"], "top_5_chunks": top5})

# Сохранить результаты
Path("results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("results.json saved")
```

### 4. Запустить score.py

```bash
python score.py --predictions results.json --questions eval_questions.json
```

Пример вывода:

```
=== CodeLens RAG -- Scoring ===

Questions evaluated: 15
Mean Precision@5: 0.453

By difficulty:
  easy     0.640 (5 questions)
  medium   0.371 (7 questions)
  hard     0.267 (3 questions)

By language:
  ru: 0.475 (8 questions)
  en: 0.429 (7 questions)

Per-question detail:
  q_01 [easy, ru] -- 2/2 expected in top-5 -> 1.00
  ...

Total score: 0.453
```

### 5. Приложить результат к работе

Метрика из строки `Total score:` — это ваш финальный Precision@5. Приложите значение и файл `results.json` к защите.

---

## FAQ

**Можно ли использовать LLM для переформулировки (expansion) запросов?**

Да. Вы можете применять query expansion, HyDE (Hypothetical Document Embeddings), цепочки мыслей или любые другие техники. Оценивается итоговая метрика, а не способ её достижения.

**Можно ли дополнительно индексировать другие репозитории помимо нашего?**

Да. Ограничений нет — можете использовать любые дополнительные данные для обучения/fine-tuning эмбеддингов. Но итоговая метрика всегда считается на нашем `eval_questions.json` против нашей кодовой базы.

**Что если мой `start_line` не совпадает с эталоном на 1-2 строки?**

Допустимый диапазон — **±2 строки**. `score.py` автоматически учитывает это при сравнении. Если вы вернули `gymhero/security.py:create_access_token:14` вместо `...12`, это засчитается.

**Что если в коде есть метод с тем же именем, что и отдельная функция?**

Формат `ClassName.method_name` решает эту проблему. Функция `create` и метод `CRUDRepository.create` — это разные chunk_id. Убедитесь, что ваш индексатор правильно различает эти случаи (см. пример с `ast` выше).

**Можно ли изменить стратегию чанкинга (не по функциям, а по строкам)?**

Технически — да, ваша система может реализовывать любую стратегию. Но `score.py` сравнивает ваши результаты с эталонами, которые указывают конкретные функции и классы. Если ваш chunk_id не соответствует формату `{path}:{name}:{line}`, скрипт не засчитает совпадение. Рекомендуем индексировать код именно по функциям/классам через AST.

**Каков формат `results.json`?**

```json
[
  {
    "question_id": "q_01",
    "top_5_chunks": [
      "gymhero/security.py:create_access_token:12",
      "gymhero/config.py:Settings:11",
      "gymhero/api/routes/auth.py:login_for_access_token:19",
      "gymhero/exceptions.py:_get_credential_exception:5",
      "gymhero/api/dependencies.py:get_token:35"
    ]
  },
  {
    "question_id": "q_02",
    "top_5_chunks": [...]
  }
]
```

Каждая запись — объект с `question_id` (строка `q_01`..`q_15`) и `top_5_chunks` — список из ровно 5 chunk_id. Дублирование в топ-5 не засчитывается повторно.

**Что если в топ-5 меньше 5 элементов?**

`score.py` выдаст предупреждение и посчитает метрику как есть. Обратите внимание: знаменатель формулы — `min(5, len(correct))`, а не фиксированная 5, поэтому меньшее число элементов в топ-5 повлияет на результат только если вы пропустили какой-то эталон.

**Q: Почему у разных вопросов разное число эталонов?**

A: Число эталонов соответствует архитектурной сложности ответа. Вопрос про конкретную функцию имеет 1 эталон. Вопрос про взаимодействие двух компонентов — 2. Вопрос про сквозной поток через несколько слоёв — 3. Скрипт `score.py` нормализует метрику: при любом числе эталонов идеальное решение даёт 1.0.

**Какой результат score.py предъявлять на защите?**

Строку `Total score:` из вывода скрипта + файл `results.json`. Жюри проверит ваш результат самостоятельным прогоном `score.py` с тем же `results.json`.

---

## Что не входит в датасет

- **Тренировочный набор** — нет. Весь `eval_questions.json` — это финальный тест.
- **Рекомендованная модель эмбеддингов** — нет. Выбор модели полностью за командой.
- **Валидационный набор** — нет отдельного валидационного сета. `sample_queries.txt` — только для демонстрации и ручной проверки системы.
- **Реализация RAG-сервиса** — нет. Команды реализуют сервис самостоятельно.
- **Оценочные критерии сверх метрики** — нет. Единственная количественная метрика — Precision@5 из `score.py`. Качественная оценка проводится жюри на защите.

---

## Технические требования к вашему решению

1. **Входные данные:** `codebase_python.zip` — распаковать, проиндексировать `.py`-файлы
2. **Формат идентификаторов:** строго `{relative_path}:{name}:{start_line}` (прямые слэши)
3. **Выход:** файл `results.json` с массивом из 15 объектов, по одному на каждый `question_id` из `eval_questions.json`
4. **Метрика:** `python score.py --predictions results.json --questions eval_questions.json`

---

*Датасет подготовлен организаторами чемпионата ГК «Ростелеком», 2026. Кодовая база gymhero распространяется по лицензии MIT — см. `gymhero/LICENSE`.*

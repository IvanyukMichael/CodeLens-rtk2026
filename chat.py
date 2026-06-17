"""CodeLens — связный LLM-ответ по найденным фрагментам (бонус §8 CLAUDE.md).

Поверх поиска: берём top-N фрагментов из search.search(), отдаём их локальной
LLM (Ollama qwen2.5-coder:7b) и просим связно ответить на вопрос, опираясь
ТОЛЬКО на показанный код. Это «RAG-ответ»: retrieval уже сделан ядром search.py,
здесь — generation.

Принципы:
  - модель и адрес Ollama переиспользуем из hyde.py (один источник правды);
  - отвечаем на языке вопроса (RU/EN), ссылаемся на фрагменты как `path:name`;
  - если в найденном коде ответа нет — модель обязана это сказать (анти-галлюцинация);
  - graceful fallback: если Ollama недоступна, ollama_chat* поднимает ошибку,
    а UI показывает сами фрагменты без LLM-ответа (демо не падает).

Запуск как CLI (быстрая проверка без Streamlit):
    python chat.py "как создаётся токен доступа"
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json

import requests

# Переиспользуем адрес и имя модели из экспериментального hyde.py (не дублируем).
from hyde import OLLAMA_URL, LLM_MODEL

MAX_CHUNKS = 5            # сколько фрагментов кладём в контекст LLM
MAX_CODE_CHARS = 1400     # обрезка очень длинного фрагмента, чтобы влезть в контекст

CHAT_SYSTEM = (
    "You are CodeLens, an assistant that answers questions about a Python codebase "
    "STRICTLY from the provided code fragments.\n"
    "Rules:\n"
    "- Answer in the SAME language as the user's question (Russian or English).\n"
    "- Use ONLY the given fragments. Do NOT invent functions, files or behaviour.\n"
    "- Cite the fragments you rely on as `path:name` (e.g. `gymhero/security.py:create_access_token`).\n"
    "- If the fragments do not contain the answer, say so honestly and do not guess.\n"
    "- Be concise: a short explanation, then the key citations."
)


def ollama_available(timeout: float = 1.5) -> bool:
    """Жива ли Ollama (для UI-индикатора и пред-проверки)."""
    try:
        requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=timeout)
        return True
    except Exception:
        return False


def build_context(results: list[dict], max_chunks: int = MAX_CHUNKS,
                  max_chars: int = MAX_CODE_CHARS) -> str:
    """Склеивает найденные фрагменты в текстовый контекст для LLM."""
    blocks = []
    for i, hit in enumerate(results[:max_chunks], 1):
        code = hit["code"]
        if len(code) > max_chars:
            code = code[:max_chars] + "\n# … (фрагмент обрезан)"
        head = f"[Фрагмент {i}] {hit['path']}:{hit['name']}  (релевантность {hit['relevance']:.0f}%)"
        blocks.append(f"{head}\n```python\n{code}\n```")
    return "\n\n".join(blocks)


def _build_prompt(question: str, results: list[dict]) -> str:
    context = build_context(results)
    return (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Найденные фрагменты кода:\n{context}\n\n"
        f"Ответь на вопрос, опираясь только на эти фрагменты."
    )


def ollama_chat_stream(question: str, results: list[dict], temperature: float = 0.0):
    """Генератор: отдаёт ответ LLM токенами по мере поступления (для st.write_stream).

    Бросает requests-исключение, если Ollama недоступна — вызывающий решает,
    как деградировать (UI показывает фрагменты без ответа)."""
    if not results:
        yield "Нет найденных фрагментов — нечего обобщать."
        return
    resp = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "system": CHAT_SYSTEM,
        "prompt": _build_prompt(question, results),
        "stream": True,
        "options": {"temperature": temperature, "num_predict": 512},
    }, stream=True, timeout=180)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("response", "")
        if token:
            yield token
        if chunk.get("done"):
            break


def ollama_chat(question: str, results: list[dict], temperature: float = 0.0) -> str:
    """Нестриминговый вариант: собирает полный ответ строкой."""
    return "".join(ollama_chat_stream(question, results, temperature=temperature))


def _cli() -> None:
    import search

    query = " ".join(sys.argv[1:]) or "как создаётся токен доступа"
    res = search.search(query, top_k=MAX_CHUNKS, use_hyde=True)
    print(f"Запрос: {query!r}  (найдено {len(res['results'])} фрагментов)")
    print("-" * 78)
    if not ollama_available():
        print(f"[!] Ollama ({LLM_MODEL}) недоступна — показываю только фрагменты:")
        for i, h in enumerate(res["results"], 1):
            print(f"{i}. [{h['relevance']:5.1f}%] {h['path']}:{h['name']}")
        return
    print("Ответ LLM:\n")
    for tok in ollama_chat_stream(query, res["results"]):
        print(tok, end="", flush=True)
    print()


if __name__ == "__main__":
    _cli()

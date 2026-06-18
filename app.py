"""CodeLens — Streamlit UI (вторая обязательная команда продукта).

    streamlit run app.py

Две страницы (вкладки):
  🔎 Поиск    — запрос RU/EN, тумблер HyDE, top-K, карточки с подсветкой кода,
                relevance%, latency, обработка «негативных» запросов.
  📊 Метрики  — живой прогон Precision@5 (официальный score) на 15 вопросах
                (опц. на расширенных 71) со срезами easy/medium/hard и ru/en.

Поиск использует замороженное ядро search.py (bge-m3 + enriched + HyDE-mix).
Модель bge-m3 и клиент ChromaDB кэшируются через @st.cache_resource.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import chat
import search
from score import score_question

OFFICIAL_EVAL = Path("eval_questions.json")
COMBINED_EVAL = Path("eval/eval_combined.json")

st.set_page_config(page_title="CodeLens — семантический поиск по коду",
                   page_icon="🔍", layout="wide")


# ─────────────────────────── ресурсы (кэш на процесс) ───────────────────────────
@st.cache_resource(show_spinner="Загрузка bge-m3 и ChromaDB…")
def get_engine():
    """Грузит модель и коллекцию один раз (прогревает синглтоны search.py)."""
    return search.warmup()


@st.cache_data(ttl=30, show_spinner=False)
def ollama_up() -> bool:
    # Адрес берём из chat/hyde (настраивается через env OLLAMA_URL для Docker).
    return chat.ollama_available()


def db_ready() -> bool:
    try:
        get_engine()
        return True
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def corpus_overview() -> dict:
    """Состав индексируемой базы прямо из ChromaDB (для вкладки Метрик).
    Читает метаданные коллекции и считает файлы/чанки по источникам и языкам.
    Фолбэк по `path` — чтобы корректно работать и на старой базе без полей
    source/lang (тогда source = первый компонент пути, lang = python)."""
    from collections import defaultdict
    col = search.get_collection()
    metas = col.get(include=["metadatas"]).get("metadatas") or []

    src_files, src_chunks, lang_chunks, all_files = (
        defaultdict(set), defaultdict(int), defaultdict(int), set())
    for m in metas:
        path = m.get("path", "")
        src = m.get("source") or (path.split("/", 1)[0] if path else "?")
        lang = m.get("lang") or "python"
        src_files[src].add(path)
        src_chunks[src] += 1
        lang_chunks[lang] += 1
        all_files.add(path)

    by_source = sorted(([s, len(src_files[s]), src_chunks[s]] for s in src_chunks),
                       key=lambda r: -r[2])
    by_lang = sorted(([lng, lang_chunks[lng]] for lng in lang_chunks),
                     key=lambda r: -r[1])
    return {"n_files": len(all_files), "n_chunks": len(metas),
            "n_langs": len(lang_chunks), "by_source": by_source, "by_lang": by_lang}


# ─────────────────────────────── eval-хелперы ───────────────────────────────────
def run_eval(questions: list[dict], use_hyde: bool, progress=None) -> list[dict]:
    per_q = []
    n = len(questions)
    for i, q in enumerate(questions):
        r = search.search(q["query"], top_k=5, use_hyde=use_hyde)
        top5 = [h["chunk_id"] for h in r["results"]]
        per_q.append({**q, "score": score_question(top5, q["correct_chunk_ids"])})
        if progress is not None:
            progress.progress((i + 1) / n, text=f"Прогон eval… {i + 1}/{n}")
    return per_q


def aggregate(per_q: list[dict]) -> pd.DataFrame:
    def row(label, pred):
        vals = [r["score"] for r in per_q if pred(r)]
        if not vals:
            return None
        return {"Срез": label, "P@5": sum(vals) / len(vals), "Вопросов": len(vals)}

    rows = [row("ВСЕГО", lambda r: True)]
    for d in ("easy", "medium", "hard"):
        rows.append(row(f"difficulty: {d}", lambda r, d=d: r.get("difficulty") == d))
    for lang in ("ru", "en"):
        rows.append(row(f"язык: {lang}", lambda r, lang=lang: r.get("language") == lang))
    if any("subset" in r for r in per_q):
        n_off = sum(1 for r in per_q if r.get("subset") == "official")
        n_ext = sum(1 for r in per_q if r.get("subset") == "extended")
        rows.append(row(f"официальные ({n_off})", lambda r: r.get("subset") == "official"))
        rows.append(row(f"расширенные ({n_ext})", lambda r: r.get("subset") == "extended"))
    return pd.DataFrame([r for r in rows if r is not None])


# ─────────────────────────────────── сайдбар ────────────────────────────────────
with st.sidebar:
    st.title("🔍 CodeLens")
    st.caption("Семантический поиск по Python-коду · bge-m3 + enriched + HyDE-mix")

    st.subheader("Параметры")
    _ollama = ollama_up()
    use_hyde = st.toggle(
        "HyDE", value=_ollama,
        help="Гипотетический код через Ollama (qwen2.5-coder:7b) усредняется с "
             "эмбеддингом запроса. Требует запущенной Ollama.")
    if use_hyde and not _ollama:
        st.warning("Ollama не отвечает — поиск пойдёт без HyDE (graceful fallback).")
    elif _ollama:
        st.caption("🟢 Ollama доступна")
    else:
        st.caption("🔴 Ollama недоступна")

    top_k = st.slider("top-K результатов", 1, 10, 5)
    st.divider()
    st.caption(f"Порог «не найдено»: relevance < {search.NEGATIVE_THRESHOLD:.0f}%")


if not db_ready():
    st.error(
        "База не найдена. Сначала постройте индекс:\n\n"
        "```\npython index.py gymhero\n```\n\n"
        "затем перезапустите `streamlit run app.py`.")
    st.stop()


# ──────────────────────────────── рендер поиска ─────────────────────────────────
def render_results(res: dict) -> None:
    results = res["results"]
    if res.get("warning"):
        st.warning("⚠️ " + res["warning"])

    if not results:
        st.info("Ничего не найдено.")
        return

    best = max(h["relevance"] for h in results)
    if best < search.NEGATIVE_THRESHOLD:
        st.error(
            f"🚫 Релевантного кода не найдено (лучшая релевантность {best:.0f}% < "
            f"{search.NEGATIVE_THRESHOLD:.0f}%). Возможно, такой функциональности нет "
            f"в базе. Ближайшие результаты — ниже.")

    for i, hit in enumerate(results, 1):
        with st.container(border=True):
            top = st.columns([5, 2])
            with top[0]:
                icon = "🏛️" if hit["kind"] == "class" else "🔧"
                st.markdown(
                    f"**{i}. `{hit['path']}` → `{hit['name']}`**")
                st.caption(
                    f"{icon} {hit['kind']} · строки {hit['start']}–{hit['end']} · "
                    f"`{hit['chunk_id']}`")
            with top[1]:
                st.progress(min(1.0, hit["relevance"] / 100.0),
                            text=f"relevance {hit['relevance']:.0f}%")
            if hit.get("docstring"):
                st.caption("📝 " + hit["docstring"].splitlines()[0])
            st.code(hit["code"], language="python")

    # latency — под результатами (требование ТЗ ≤ 3 c)
    st.divider()
    lat = res["latency"]
    total_s = lat["total_ms"] / 1000.0
    c = st.columns(4)
    c[0].metric("HyDE", f"{lat['hyde_ms']:.0f} мс",
                help="cache" if res.get("hyde_from_cache") else None)
    c[1].metric("Эмбеддинг", f"{lat['embed_ms']:.0f} мс")
    c[2].metric("Поиск", f"{lat['search_ms']:.0f} мс")
    c[3].metric("Итого", f"{total_s:.2f} с")
    if total_s <= 3.0:
        st.success(f"✅ Латентность {total_s:.2f} с ≤ 3 с (требование ТЗ)")
    else:
        st.warning(f"⚠️ Латентность {total_s:.2f} с > 3 с")


tab_search, tab_chat, tab_metrics = st.tabs(["🔎 Поиск", "💬 Ответ (LLM)", "📊 Метрики"])

with tab_search:
    st.subheader("Поиск по коду")
    with st.form("search_form"):
        query = st.text_input(
            "Запрос (RU или EN)",
            placeholder="напр.: как создаётся токен доступа  /  where is the superuser checked")
        submitted = st.form_submit_button("🔎 Найти", type="primary")

    if submitted and query.strip():
        with st.spinner("Поиск…"):
            res = search.search(query, top_k=top_k, use_hyde=use_hyde)
        st.session_state["last_search"] = res
    elif submitted:
        st.info("Введите запрос.")

    if "last_search" in st.session_state:
        render_results(st.session_state["last_search"])
    else:
        st.caption("Примеры: «как хешируется пароль», «создание пользователя в БД», "
                   "«rate limiting» (негативный — такого в базе нет).")


def render_fragments_compact(results: list[dict]) -> None:
    """Компактный список фрагментов под LLM-ответом (источники)."""
    for i, hit in enumerate(results, 1):
        with st.expander(
            f"{i}. {hit['path']}:{hit['name']}  ·  relevance {hit['relevance']:.0f}%",
            expanded=(i == 1)):
            st.code(hit["code"], language="python")


with tab_chat:
    st.subheader("Связный ответ по найденному коду")
    st.caption("RAG-ответ: ядро находит релевантные фрагменты, затем локальная LLM "
               "(Ollama · qwen2.5-coder:7b) обобщает их в связный ответ со ссылками на "
               "`path:name`. Отвечает строго по найденному коду — если ответа в нём "
               "нет, честно об этом сообщает.")

    _ollama_chat = ollama_up()
    if not _ollama_chat:
        st.warning("🔴 Ollama недоступна — LLM-ответ отключён. Будут показаны только "
                   "найденные фрагменты. Запустите Ollama и `ollama pull "
                   "qwen2.5-coder:7b`, чтобы включить генерацию ответа.")

    with st.form("chat_form"):
        chat_query = st.text_input(
            "Вопрос (RU или EN)",
            placeholder="напр.: как проверяется, что пользователь — суперпользователь?")
        chat_submitted = st.form_submit_button("💬 Ответить", type="primary")

    if chat_submitted and chat_query.strip():
        with st.spinner("Поиск фрагментов…"):
            cres = search.search(chat_query, top_k=chat.MAX_CHUNKS, use_hyde=use_hyde)
        results = cres["results"]
        if cres.get("warning"):
            st.info("⚠️ " + cres["warning"])

        if not results:
            st.info("Ничего не найдено.")
        else:
            best = max(h["relevance"] for h in results)
            if best < search.NEGATIVE_THRESHOLD:
                st.error(
                    f"🚫 Релевантного кода не найдено (лучшая релевантность {best:.0f}% < "
                    f"{search.NEGATIVE_THRESHOLD:.0f}%). Вероятно, такой функциональности "
                    f"нет в базе — LLM-ответ может быть неинформативным.")

            if _ollama_chat:
                st.markdown("#### 🤖 Ответ")
                try:
                    with st.spinner("Генерация ответа (Ollama)…"):
                        st.write_stream(chat.ollama_chat_stream(chat_query, results))
                except Exception as e:
                    st.warning(f"Ollama не ответила ({type(e).__name__}). "
                               f"Показаны только фрагменты.")

            st.markdown("#### 📎 Источники (найденные фрагменты)")
            render_fragments_compact(results)
    else:
        st.caption("Введите вопрос — система найдёт фрагменты и (если есть Ollama) "
                   "соберёт из них связный ответ.")


with tab_metrics:
    st.subheader("📦 Состав индексируемой базы")
    st.caption("Живые числа из ChromaDB. Требование ТЗ — база ≥ 80 файлов. "
               "Поиск идёт по всему корпусу: gymhero (официальный) + открытые репозитории "
               "(click, rich) + JS/Java-демокорпус.")
    ov = corpus_overview()
    c = st.columns(3)
    c[0].metric("Файлов", ov["n_files"],
                help="Уникальных файлов, попавших в индекс (с ≥1 чанком; пустые "
                     "__init__.py без функций/классов сюда не входят)")
    c[1].metric("Чанков", ov["n_chunks"],
                help="Функций / классов / методов в коллекции")
    c[2].metric("Языков", ov["n_langs"],
                help="python / javascript / java")
    if ov["n_files"] >= 80:
        st.success(f"✅ {ov['n_files']} файлов ≥ 80 (требование ТЗ выполнено)")
    else:
        st.warning(f"⚠️ {ov['n_files']} файлов < 80 — пересоберите индекс: `python index.py` "
                   f"(сейчас в базе только часть корпуса).")
    ov_cols = st.columns(2)
    with ov_cols[0]:
        st.caption("По источникам")
        st.dataframe(
            pd.DataFrame(ov["by_source"], columns=["Источник", "Файлов", "Чанков"]),
            use_container_width=True, hide_index=True)
    with ov_cols[1]:
        st.caption("По языкам")
        st.dataframe(
            pd.DataFrame(ov["by_lang"], columns=["Язык", "Чанков"]),
            use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Precision@5 (официальный score.py)")
    st.caption("Живой прогон: для каждого вопроса делаем поиск текущим ядром и "
               "считаем P@5 тем же score_question, что у организаторов. "
               f"Режим HyDE берётся из сайдбара (сейчас: {'вкл' if use_hyde else 'выкл'}).")

    def _count(path: Path, default: int) -> int:
        try:
            return len(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return default

    n_off = _count(OFFICIAL_EVAL, 15)
    n_comb = _count(COMBINED_EVAL, 71)
    opt_off, opt_comb = f"Официальный ({n_off})", f"Расширенный ({n_comb})"
    eval_choice = st.radio(
        "Набор вопросов", [opt_off, opt_comb],
        horizontal=True,
        help=f"Чемпионатная метрика считается на официальных {n_off}. Расширенный "
             f"набор ({n_comb}) — наш для устойчивости оценки.")

    if st.button("▶ Прогнать eval", type="primary"):
        path = OFFICIAL_EVAL if eval_choice == opt_off else COMBINED_EVAL
        if not path.exists():
            st.error(f"Файл вопросов не найден: {path}")
        else:
            questions = json.loads(path.read_text(encoding="utf-8"))
            if use_hyde and not ollama_up():
                st.info("Ollama недоступна — eval идёт без HyDE (fallback).")
            prog = st.progress(0.0, text="Прогон eval…")
            per_q = run_eval(questions, use_hyde=use_hyde, progress=prog)
            prog.empty()
            st.session_state["eval_result"] = {
                "df": aggregate(per_q),
                "set": eval_choice,
                "hyde": use_hyde,
                "n": len(per_q),
            }

    if "eval_result" in st.session_state:
        er = st.session_state["eval_result"]
        df = er["df"]
        total = float(df.loc[df["Срез"] == "ВСЕГО", "P@5"].iloc[0])
        m = st.columns(3)
        m[0].metric("Total Precision@5", f"{total:.3f}")
        m[1].metric("Набор", er["set"])
        m[2].metric("HyDE", "вкл" if er["hyde"] else "выкл")
        thr = 0.60
        if total >= 0.85:
            st.success(f"🎯 P@5 = {total:.3f} ≥ 0.85 (внутренняя цель)")
        elif total >= thr:
            st.info(f"P@5 = {total:.3f} ≥ {thr:.2f} (порог ТЗ)")
        else:
            st.warning(f"P@5 = {total:.3f} < {thr:.2f}")
        st.dataframe(
            df.style.format({"P@5": "{:.3f}"}),
            use_container_width=True, hide_index=True)

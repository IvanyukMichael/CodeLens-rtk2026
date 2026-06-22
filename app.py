"""CodeLens — Streamlit UI (вторая обязательная команда продукта).

    streamlit run app.py

Две страницы (вкладки):
  🔎 Поиск    — запрос RU/EN, тумблер HyDE, top-K, карточки с подсветкой кода,
                relevance%, latency, обработка «негативных» запросов.
  📊 Метрики  — живой прогон Precision@5 (официальный score) на 15 вопросах
                (опц. на combined-269 как robustness) со срезами easy/medium/hard и ru/en.

Поиск использует замороженное ядро search.py (bge-m3 + enriched + HyDE-mix).
Модель bge-m3 и клиент ChromaDB кэшируются через @st.cache_resource.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import add_repo
import chat
import search
from score import score_question

OFFICIAL_EVAL = Path("eval_questions.json")
COMBINED_EVAL = Path("eval/eval_combined.json")
RESULTS_DIR = Path("results")

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
    """Состав ВСЕЙ индексируемой базы прямо из ChromaDB (для вкладки Метрик).
    Агрегирует ОБЕ коллекции (codelens_gymhero + codelens_extra), чтобы показать
    требование ТЗ «≥80 файлов» по всей базе. Метрика P@5 при этом считается
    отдельно — строго по gymhero-коллекции (см. ниже). Фолбэк по `path` — чтобы
    работать и на старой базе без полей source/lang."""
    from collections import defaultdict
    metas = []
    for name in (search.GYMHERO_COLLECTION, search.EXTRA_COLLECTION):
        try:
            col = search.get_collection(name)
        except Exception:
            continue                              # коллекция не построена — пропускаем
        metas += col.get(include=["metadatas"]).get("metadatas") or []

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
        # ОФИЦИАЛЬНАЯ метрика всегда по gymhero-коллекции (эталонные chunk_id
        # размечены только по gymhero) — не зависит от селектора источника в UI.
        r = search.search(q["query"], top_k=5, use_hyde=use_hyde, source="gymhero")
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

    # Источник поиска: gymhero (официальная метрика 0.800) vs вся база (≥80 файлов).
    SRC_GYM = "Только gymhero (офиц. 0.800)"
    SRC_ALL = "Вся база ≥80 файлов"
    _src_opts = [SRC_GYM]
    if "all" in search.available_sources():
        _src_opts.append(SRC_ALL)
    src_label = st.radio(
        "Область поиска", _src_opts,
        help="«Только gymhero» — официальный корпус, по нему считается метрика "
             "(P@5=0.800). «Вся база» — gymhero + открытые репозитории (click/rich) "
             "+ JS/Java (≥80 файлов ТЗ), мультипроектный поиск. Метрика на вкладке "
             "«Метрики» всегда считается по gymhero независимо от этого выбора.")
    search_source = "all" if src_label == SRC_ALL else "gymhero"
    if len(_src_opts) == 1:
        st.caption("ℹ️ Доп. коллекция не построена — доступен только gymhero. "
                   "Постройте всю базу: `python index.py`")

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
                badge = f" · 📦 {hit['source']}" if hit.get("source") else ""
                st.markdown(
                    f"**{i}. `{hit['path']}` → `{hit['name']}`**")
                st.caption(
                    f"{icon} {hit['kind']} · строки {hit['start']}–{hit['end']}{badge} · "
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


tab_search, tab_chat, tab_metrics, tab_results, tab_repos = st.tabs(
    ["🔎 Поиск", "💬 Ответ (LLM)", "📊 Метрики", "📈 Результаты исследования",
     "📦 Репозитории"])

with tab_search:
    st.subheader("Поиск по коду")
    with st.form("search_form"):
        query = st.text_input(
            "Запрос (RU или EN)",
            placeholder="напр.: как создаётся токен доступа  /  where is the superuser checked")
        submitted = st.form_submit_button("🔎 Найти", type="primary")

    if submitted and query.strip():
        with st.spinner("Поиск…"):
            res = search.search(query, top_k=top_k, use_hyde=use_hyde,
                                source=search_source)
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
            cres = search.search(chat_query, top_k=chat.MAX_CHUNKS, use_hyde=use_hyde,
                                 source=search_source)
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
    st.caption("Живые числа из ChromaDB по ОБЕИМ коллекциям. Требование ТЗ — база "
               "≥ 80 файлов. База разнесена на две изолированные коллекции: "
               "`codelens_gymhero` (официальный корпус, по нему метрика) и "
               "`codelens_extra` (click/rich + JS/Java — размер базы и "
               "мультипроектный поиск в режиме «Вся база»).")
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
               "считаем P@5 тем же score_question, что у организаторов. Метрика "
               "считается **строго по gymhero-коллекции** (эталонные chunk_id "
               "размечены только по gymhero) — независимо от выбора «Область поиска» "
               "в сайдбаре. Поэтому P@5 не зависит от размера общей базы. "
               f"Режим HyDE берётся из сайдбара (сейчас: {'вкл' if use_hyde else 'выкл'}).")

    def _count(path: Path, default: int) -> int:
        try:
            return len(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return default

    n_off = _count(OFFICIAL_EVAL, 15)
    n_comb = _count(COMBINED_EVAL, 269)
    opt_off, opt_comb = f"Официальный ({n_off})", f"Combined ({n_comb}) · robustness"
    eval_choice = st.radio(
        "Набор вопросов", [opt_off, opt_comb],
        horizontal=True,
        help=f"Headline-метрика — официальные {n_off} (человеческая разметка). Combined "
             f"({n_comb}) = 15 + синтетические — только robustness-проверка агрегата: "
             f"LLM-вопросы инфлируют lexical-методы, поэтому в срезы их не берём.")

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


with tab_results:
    st.subheader("📈 Результаты исследования (ablation)")
    st.caption("Готовые таблицы из `results/*.csv` и графики. Read-only — ничего не считается "
               "на лету. Каждая таблица помечена набором eval, на котором получена.")

    def _fmt_df(df: pd.DataFrame):
        floatcols = [c for c in df.columns if df[c].dtype.kind == "f"]
        return df.style.format({c: "{:.3f}" for c in floatcols})

    def show_table(fname: str, title: str, eval_label: str, script: str, note=None):
        path = RESULTS_DIR / fname
        st.markdown(f"**{title}**  ·  набор: _{eval_label}_")
        if note:
            st.caption(note)
        if path.exists():
            try:
                st.dataframe(_fmt_df(pd.read_csv(path)),
                             use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Не удалось прочитать {fname}: {e}")
        else:
            st.info(f"Ещё не запущено — нет `{path}`. Сгенерировать: `python {script}`")
        st.divider()

    h = st.columns(3)
    h[0].metric("P@5 · офиц. 15 (headline)", "0.800")
    h[1].metric("P@5 · combined-269", "0.890", help="robustness-агрегат (15 + 254 синтетических); "
                "синтетика инфлирует lexical-методы, headline и срезы — на официальных 15")
    h[2].metric("Прод-конфигурация", "bge-m3 + enriched + HyDE")
    st.caption("Headline — 0.800 на официальных 15 (человеческая разметка). Combined-269 (0.890) — "
               "только robustness. Полная методология, CI и выводы — в REPORT.md.")
    st.divider()

    st.markdown("### Путь модели и общая ablation")
    show_table("ablation_models.csv", "Bake-off моделей (4 модели × raw/enriched)",
               "официальные 15", "eval_runner.py --all")
    show_table("ablation_full.csv", "Полная цепочка + подмножества + бутстрап значимости",
               "269 (15 офиц. + 254 расш.)", "ablation_full.py")

    st.markdown("### Рычаги ретрива (принятые и отклонённые)")
    show_table("ablation_hyde.csv", "HyDE: base / pure / mix + α-свип  —  ✅ принят",
               "официальные 15", "hyde.py")
    show_table("ablation_rerank.csv", "Reranker bge-reranker-v2-m3  —  ❌ отклонён (в пределах шума)",
               "официальные 15", "rerank.py")
    show_table("ablation_hybrid.csv", "BM25-гибрид (RRF)  —  ❌ отклонён (оверфит на 15-q)",
               "официальные 15", "hybrid.py")
    show_table("ablation_hyde_multi.csv", "Multi-HyDE ×3  —  ❌ отклонён (нет прироста)",
               "официальные 15", "hyde_multi.py")

    st.markdown("### Приоритет 3.2 — новые рычаги (прогон на GPU / с ключами)")
    show_table("ablation_rerank_jina.csv", "jina-reranker-v2 (код-tuned cross-encoder)",
               "170 (15 + 155)", "exp_rerank_jina.py",
               note="Запускать на CUDA. Принять/отклонить — по доверительному интервалу на 170.")
    show_table("ablation_api_embedders.csv",
               "API-эмбеддеры: Cohere embed-v4 / Voyage voyage-code-3",
               "170 (15 + 155)", "exp_api_embedders.py",
               note="Нужны ключи COHERE_API_KEY / VOYAGE_API_KEY. «Потолок» для отчёта, "
                    "прод остаётся на локальной bge-m3.")

    st.markdown("### Графики")
    figs = [
        ("fig_model_bakeoff.png", "Bake-off моделей"),
        ("fig_p5_by_config.png", "P@5 по конфигурациям"),
        ("fig_ru_vs_en.png", "P@5: русский vs английский"),
    ]
    cols = st.columns(len(figs))
    shown = False
    for col, (fname, cap) in zip(cols, figs):
        p = RESULTS_DIR / fname
        if p.exists():
            col.image(str(p), caption=cap, use_container_width=True)
            shown = True
        else:
            col.info(f"нет {fname}")
    if not shown:
        st.caption("Графики строятся в analysis.ipynb → results/*.png.")


# ─────────────────────────── вкладка «Репозитории» ──────────────────────────────
# Песочница: управление пользовательскими репозиториями + поиск по выбранному репо.
# Полностью изолирована от gymhero — пишет и ищет СТРОГО в codelens_extra
# (source=<owner__repo>, origin="user"). Официальная метрика (вкладки Поиск/Метрики)
# не затрагивается: тот же search()-движок, но с source="extra" + where по репо.
_STAGE_LABELS = {
    "clone": "Клонирование репозитория…", "limits": "Проверка лимитов…",
    "parse": "Парсинг .py (AST)…", "enrich": "Обогащение чанков…",
    "embed": "Эмбеддинги (bge-m3)…", "write": "Запись в ChromaDB (codelens_extra)…",
    "done": "Готово",
}


def _index_repo_with_status(url: str, reindex: bool) -> dict:
    """Запускает индексацию репо с живым прогрессом в st.status (не «висит молча»)."""
    with st.status("Индексация репозитория…", expanded=True) as status:
        def _prog(stage: str, detail: str = "") -> None:
            label = _STAGE_LABELS.get(stage, stage)
            status.update(label=label)
            st.write(f"• {label} {detail}".rstrip())
        res = add_repo.index_repo(url, reindex=reindex, progress=_prog)
        status.update(label="Готово", state="complete")
    return res


with tab_repos:
    st.subheader("📦 Пользовательские репозитории (песочница)")
    st.caption(
        "Добавьте публичный GitHub-репозиторий — он индексируется тем же ядром "
        "(extract_chunks + enrich + bge-m3) в изолированную коллекцию "
        "`codelens_extra` с тегами `source=owner__repo`, `origin=user`. Официальная "
        "метрика gymhero (P@5 = 0.800) и вкладки «Поиск»/«Метрики» этим не "
        "затрагиваются — поиск здесь идёт строго по выбранному репозиторию. "
        "Добавленные репозитории переживают `python index.py`.")

    # ── A. Добавление ────────────────────────────────────────────────────────
    st.markdown("#### ➕ Добавить репозиторий")
    with st.form("add_repo_form"):
        repo_url = st.text_input(
            "Ссылка на GitHub-репозиторий",
            placeholder="https://github.com/owner/repo")
        reindex = st.checkbox("Переиндексировать, если уже добавлен", value=False,
                              help="Иначе повторное добавление того же репо будет "
                                   "пропущено как дубль.")
        add_submitted = st.form_submit_button("Добавить и проиндексировать",
                                              type="primary")

    if add_submitted and repo_url.strip():
        try:
            res = _index_repo_with_status(repo_url, reindex)
            if res["status"] == "duplicate":
                st.warning(
                    f"⚠️ Репозиторий **{res['repo_name']}** уже добавлен "
                    f"({res['n_chunks']} чанков). Отметьте «Переиндексировать», "
                    f"чтобы обновить, либо удалите его ниже.")
            else:
                verb = "переиндексирован" if res["status"] == "reindexed" else "добавлен"
                st.success(
                    f"✅ Репозиторий **{res['repo_name']}** {verb}: файлов "
                    f"{res.get('n_files', '?')}, чанков {res['n_chunks']} "
                    f"(способ: {res.get('fetch')}).")
            corpus_overview.clear()        # обновить «состав базы» во вкладке «Метрики»
        except add_repo.RepoError as e:    # ожидаемая проблема — понятный текст, не краш
            st.error(f"❌ {e}")
        except Exception as e:             # непредвиденное — тоже не роняем вкладку
            st.error(f"❌ Непредвиденная ошибка: {type(e).__name__}: {e}")
    elif add_submitted:
        st.info("Введите ссылку на репозиторий.")

    st.divider()

    # ── B. Список добавленных + удаление по одному ───────────────────────────
    st.markdown("#### 📚 Добавленные репозитории")
    try:
        user_repos = add_repo.list_user_repos()
    except Exception as e:
        user_repos = []
        st.warning(f"Не удалось прочитать список репозиториев: {e}")

    if not user_repos:
        st.caption("Пока ничего не добавлено. Для пакетного добавления заранее можно "
                   "положить ссылки в `repos.txt` и выполнить "
                   "`python add_repo.py --file repos.txt`.")
    else:
        for r in user_repos:
            cols = st.columns([5, 2, 2])
            cols[0].markdown(f"**{r['repo_name']}**  \n`{r['url']}`")
            cols[1].metric("Чанков", r["n_chunks"])
            if cols[2].button("🗑 Удалить", key=f"del_{r['repo_name']}",
                              help="Удаляет только чанки этого репозитория (по source)."):
                try:
                    n = add_repo.remove_repo(r["repo_name"])
                    st.success(f"Удалён **{r['repo_name']}** ({n} чанков).")
                except Exception as e:
                    st.error(f"Не удалось удалить: {e}")
                corpus_overview.clear()
                st.session_state.pop("repo_search", None)
                st.rerun()

    st.divider()

    # ── C. Поиск по выбранному репозиторию ───────────────────────────────────
    st.markdown("#### 🔍 Поиск по репозиторию")
    if not user_repos:
        st.caption("Добавьте репозиторий выше, чтобы искать по нему.")
    else:
        names = [r["repo_name"] for r in user_repos]
        sel = st.selectbox("Репозиторий для поиска", names, key="repo_select")
        repo_llm = st.toggle(
            "LLM-ответ по найденному", value=False, key="repo_llm",
            help="Собрать связный ответ локальной LLM (Ollama · qwen2.5-coder:7b) "
                 "по найденным фрагментам выбранного репозитория.")
        with st.form("repo_search_form"):
            repo_query = st.text_input(
                "Запрос (RU или EN)", key="repo_query",
                placeholder="напр.: how is configuration loaded  /  где обрабатываются ошибки")
            repo_submitted = st.form_submit_button("🔎 Найти", type="primary")

        if repo_submitted and repo_query.strip():
            with st.spinner("Поиск…"):
                rres = search.search(repo_query, top_k=top_k, use_hyde=use_hyde,
                                     source="extra", where={"source": sel})
            st.session_state["repo_search"] = {
                "res": rres, "repo": sel, "query": repo_query, "llm": repo_llm}
        elif repo_submitted:
            st.info("Введите запрос.")

        rs = st.session_state.get("repo_search")
        if rs and rs["repo"] == sel:                # не показывать выдачу от другого репо
            if rs["llm"]:
                results = rs["res"]["results"]
                if rs["res"].get("warning"):
                    st.info("⚠️ " + rs["res"]["warning"])
                if not results:
                    st.info("Ничего не найдено.")
                else:
                    best = max(h["relevance"] for h in results)
                    if best < search.NEGATIVE_THRESHOLD:
                        st.error(
                            f"🚫 Релевантного кода не найдено (лучшая релевантность "
                            f"{best:.0f}% < {search.NEGATIVE_THRESHOLD:.0f}%). "
                            f"Вероятно, такой функциональности нет в этом репозитории.")
                    if ollama_up():
                        st.markdown("#### 🤖 Ответ")
                        try:
                            with st.spinner("Генерация ответа (Ollama)…"):
                                st.write_stream(
                                    chat.ollama_chat_stream(rs["query"], results))
                        except Exception as e:
                            st.warning(f"Ollama не ответила ({type(e).__name__}). "
                                       f"Показаны только фрагменты.")
                    else:
                        st.warning("🔴 Ollama недоступна — показаны только фрагменты.")
                    st.markdown("#### 📎 Источники (найденные фрагменты)")
                    render_fragments_compact(results)
            else:
                render_results(rs["res"])

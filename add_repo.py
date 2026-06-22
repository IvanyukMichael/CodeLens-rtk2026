"""CodeLens — управление пользовательскими репозиториями (вкладка «📦 Репозитории»).

Клонирует/скачивает публичный Python-репозиторий и индексирует ТОЛЬКО его .py
в коллекцию `codelens_extra` с тегами `source=<owner__repo>` и `origin="user"`.
Переиспользует ровно то же ядро, что и основной пайплайн (extract_chunks +
enrich + bge-m3 через index.embed_docs), а не дублирует его.

═══ ИНВАРИАНТ ИЗОЛЯЦИИ gymhero (три барьера) ════════════════════════════════
  1) Запись идёт СТРОГО в EXTRA_COLLECTION; перед записью assert, что это не
     gymhero-коллекция (_write_chunks_to_extra). Имя gymhero-коллекции тут не
     используется как цель записи в принципе.
  2) Переиспользуются только переносимые куски index.py (embed_docs, _source_of,
     константы) — НЕ build_index, который умеет писать в любую коллекцию.
  3) Официальная метрика P@5 считается по source="gymhero" (predict.py, вкладка
     «Метрики») и не зависит ни от extra, ни от пользовательских репозиториев.

origin="user" => эти чанки ПЕРЕЖИВАЮТ `python index.py`: ребилд встроенного
extra-корпуса удаляет только чанки с origin!="user" (см. index._reset_builtin).
Пользовательские репозитории управляются исключительно отсюда и из вкладки.

CLI:
    python add_repo.py https://github.com/owner/repo
    python add_repo.py --file repos.txt
    python add_repo.py https://github.com/owner/repo --reindex
    python add_repo.py --list
    python add_repo.py --remove owner__repo
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sslfix  # noqa: F401  — чинит битый SSL_CERT_FILE до сетевых вызовов (git/zip/HF)

import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Переиспользуем эмбеддинг с кэшем, source-логику и константы из index.py
# (импорт index также выполняет его sslfix и настройку окружения HF).
import index
from enrich import build_enriched_map
from extract_chunks import extract_repo

REGISTRY = Path("repos_registry.json")          # реестр user-репо (источник правды для UI/дедупа)
REPOS_FILE = Path("repos.txt")                  # пакетный список ссылок (# — комментарии)

MAX_PY_FILES = 500                              # потолок на число .py
MAX_REPO_MB = 50                               # потолок на суммарный объём .py
CLONE_TIMEOUT = 180                            # сек на git clone


class RepoError(Exception):
    """Понятная для UI ошибка: битая/приватная ссылка, нет сети, нет .py, лимиты."""


# ─────────────────────────────── реестр ─────────────────────────────────────
def _load_registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────── ссылка ─────────────────────────────────────
def normalize_repo_url(url: str) -> tuple[str, str]:
    """Из любой формы ссылки → (clone_url с .git, repo_name='owner__repo').

    Поддерживает https://github.com/owner/repo, тот же с .git и/или trailing
    slash, а также git@github.com:owner/repo(.git). repo_name снабжён префиксом
    владельца — это namespace для source-тега и chunk_id, исключающий коллизии
    между разными user-репо и со встроенным корпусом (click/rich)."""
    u = (url or "").strip().rstrip("/")
    if not u:
        raise RepoError("Пустая ссылка.")
    m = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", u)
    if not m:
        m = re.match(
            r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", u)
    if not m:
        raise RepoError(
            f"Не похоже на ссылку GitHub-репозитория: {url!r}. "
            f"Пример: https://github.com/owner/repo")
    owner, repo = m["owner"], m["repo"]
    return f"https://github.com/{owner}/{repo}.git", f"{owner}__{repo}"


# ─────────────────────────── получение кода ─────────────────────────────────
def _git_clone(clone_url: str, dest: Path) -> None:
    """git clone --depth 1 --single-branch (без истории). Бросает RepoError."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", clone_url, str(dest)],
            check=True, capture_output=True, text=True, timeout=CLONE_TIMEOUT)
    except FileNotFoundError:
        raise RepoError("git не найден в PATH.")
    except subprocess.TimeoutExpired:
        raise RepoError(f"Превышено время клонирования ({CLONE_TIMEOUT} c).")
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip().splitlines()
        raise RepoError(f"git clone не удался: {msg[-1] if msg else 'неизвестная ошибка'}")


def _zip_download(clone_url: str, dest: Path) -> None:
    """Fallback: скачать zip-снапшот (codeload) и распаковать содержимое в dest.
    Пробует ветки main и master."""
    m = re.match(r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)\.git", clone_url)
    owner, repo = m["owner"], m["repo"]
    last_err = None
    for branch in ("main", "master"):
        scratch = Path(tempfile.mkdtemp(prefix="codelens_zip_"))
        try:
            zurl = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
            zpath = scratch / "repo.zip"
            urllib.request.urlretrieve(zurl, zpath)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(scratch / "ex")
            inner = next((p for p in (scratch / "ex").iterdir() if p.is_dir()), None)
            if inner is None:
                raise RepoError("Пустой архив репозитория.")
            shutil.move(str(inner), str(dest))
            return
        except RepoError:
            raise
        except Exception as e:                       # 404 ветки / сеть — пробуем другую
            last_err = e
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    raise RepoError(f"Не удалось скачать zip-снапшот (main/master): {last_err}")


def fetch_repo(clone_url: str, dest: Path) -> str:
    """Кладёт СОДЕРЖИМОЕ репозитория в dest. Сначала git, затем zip-фолбэк.
    Возвращает способ ('git' | 'zip')."""
    try:
        _git_clone(clone_url, dest)
        shutil.rmtree(dest / ".git", ignore_errors=True)   # история не нужна (лимиты/чистота)
        return "git"
    except RepoError:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        _zip_download(clone_url, dest)
        return "zip"


def _check_limits(repo_dir: Path) -> tuple[int, float]:
    """Проверяет потолки до индексации. Возвращает (n_py_files, total_mb)."""
    py = list(repo_dir.rglob("*.py"))
    if not py:
        raise RepoError("В репозитории нет .py файлов — нечего индексировать.")
    if len(py) > MAX_PY_FILES:
        raise RepoError(
            f"Слишком много .py файлов: {len(py)} > лимит {MAX_PY_FILES}.")
    total_mb = sum(p.stat().st_size for p in py) / (1024 * 1024)
    if total_mb > MAX_REPO_MB:
        raise RepoError(
            f"Слишком большой объём .py: {total_mb:.1f} МБ > лимит {MAX_REPO_MB} МБ.")
    return len(py), total_mb


# ───────────────────────── запись в коллекцию ───────────────────────────────
def _write_chunks_to_extra(chunks: list[dict], enriched: dict, emb, source: str) -> int:
    """Аддитивная запись чанков в codelens_extra. БАРЬЕР #1: пишем строго в
    extra и никогда в gymhero. Перед записью удаляем прежние чанки ЭТОГО же
    репо (по source) — для реиндекса/дедупа, не трогая чужие user-репо и
    встроенный корпус."""
    name = index.EXTRA_COLLECTION
    if name == index.GYMHERO_COLLECTION:                       # барьер: цель ≠ gymhero
        raise RuntimeError("Инвариант изоляции нарушен: цель записи — gymhero.")
    if source == "gymhero":                                    # барьер: source ≠ gymhero
        raise RuntimeError("source='gymhero' запрещён для пользовательских репо.")

    import chromadb
    index.DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index.DB_DIR))
    col = client.get_or_create_collection(                     # НЕ delete_collection!
        name=name, metadata={"hnsw:space": "cosine", "model": index.MODEL_NAME})
    col.delete(where={"source": source})                      # реиндекс/дедуп только этого репо

    ids = [c["chunk_id"] for c in chunks]
    docs = [enriched[c["chunk_id"]] for c in chunks]
    metadatas = [{
        "path": c["path"], "name": c["name"], "kind": c["kind"],
        "start_line": c["start"], "end_line": c["end"],
        "docstring": c["docstring"], "code": c["code"],
        "enriched_text": enriched[c["chunk_id"]],
        "source": source,                                     # owner__repo (фильтр/удаление)
        "origin": "user",                                     # переживает ребилд index.py
        "lang": "python",
    } for c in chunks]

    for s in range(0, len(ids), index.ADD_BATCH):
        e = s + index.ADD_BATCH
        col.add(ids=ids[s:e], embeddings=emb[s:e].tolist(),
                documents=docs[s:e], metadatas=metadatas[s:e])
    return len(ids)


# ─────────────────────────── основная операция ──────────────────────────────
def index_repo(url: str, reindex: bool = False, progress=None) -> dict:
    """Добавить/переиндексировать репозиторий. Один путь и для UI, и для CLI,
    и для repos.txt.

    progress — опц. callback(stage, detail). Стадии: clone, limits, parse,
    enrich, embed, write, done. Возвращает dict со status:
      "duplicate"  — уже в реестре и reindex=False (ничего не делали);
      "added"      — новый;
      "reindexed"  — был и reindex=True.
    Любая ожидаемая проблема — RepoError (UI показывает текст, не падает)."""
    def _p(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)

    clone_url, repo_name = normalize_repo_url(url)
    reg = _load_registry()
    was_present = repo_name in reg
    if was_present and not reindex:
        return {"status": "duplicate", "repo_name": repo_name,
                "n_chunks": reg[repo_name].get("n_chunks", 0),
                "url": reg[repo_name].get("url", clone_url)}

    repo_parent = Path(tempfile.mkdtemp(prefix="codelens_repo_"))
    repo_dir = repo_parent / repo_name              # rel-путь чанков = "<repo_name>/..."
    try:
        _p("clone", clone_url)
        method = fetch_repo(clone_url, repo_dir)

        _p("limits")
        n_files, total_mb = _check_limits(repo_dir)

        _p("parse", f"{n_files} файлов")
        chunks = extract_repo(repo_parent)          # repo_root = parent => source=repo_name
        if not chunks:
            raise RepoError("Не удалось извлечь ни одного чанка из .py файлов.")
        for c in chunks:                            # подстраховка namespacing (не должно срабатывать)
            if index._source_of(c["path"]) != repo_name:
                raise RepoError("Внутренняя ошибка: source чанка не совпал с repo_name.")

        _p("enrich")
        enriched = build_enriched_map(chunks)

        _p("embed", f"{len(chunks)} чанков")
        docs = [enriched[c["chunk_id"]] for c in chunks]
        emb, _model = index.embed_docs(docs)        # тот же bge-m3 + кэш по содержимому

        _p("write")
        n_written = _write_chunks_to_extra(chunks, enriched, emb, repo_name)

        reg[repo_name] = {
            "url": clone_url, "n_files": n_files, "n_chunks": n_written,
            "size_mb": round(total_mb, 2), "fetch": method,
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save_registry(reg)
        _p("done")
        return {"status": "reindexed" if was_present else "added",
                "repo_name": repo_name, "n_files": n_files, "n_chunks": n_written,
                "url": clone_url, "fetch": method}
    finally:
        shutil.rmtree(repo_parent, ignore_errors=True)


# ─────────────────────────── список / удаление ──────────────────────────────
def _extra_source_counts(origin: str | None = None) -> dict:
    """{source: n_chunks} из codelens_extra (опц. фильтр по origin). {} если
    коллекции нет."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(index.DB_DIR))
        col = client.get_collection(index.EXTRA_COLLECTION)
    except Exception:
        return {}
    where = {"origin": origin} if origin else None
    got = col.get(where=where, include=["metadatas"])
    counts: dict = {}
    for m in got.get("metadatas") or []:
        s = (m or {}).get("source", "?")
        counts[s] = counts.get(s, 0) + 1
    return counts


def list_user_repos() -> list[dict]:
    """Пользовательские репо из реестра со счётчиком чанков из коллекции
    (origin='user'). Встроенный корпус (click/rich/java/js) сюда НЕ попадает —
    он управляется index.py, а не вкладкой."""
    reg = _load_registry()
    counts = _extra_source_counts(origin="user")
    out = [{
        "repo_name": name,
        "url": info.get("url", ""),
        "n_chunks": counts.get(name, info.get("n_chunks", 0)),
        "added_at": info.get("added_at", ""),
    } for name, info in reg.items()]
    return sorted(out, key=lambda r: r["repo_name"])


def remove_repo(repo_name: str) -> int:
    """Удаляет ВСЕ чанки репо по source-тегу и убирает его из реестра.
    Возвращает число удалённых чанков. Встроенный корпус и чужие репо не трогает."""
    import chromadb
    n = 0
    try:
        client = chromadb.PersistentClient(path=str(index.DB_DIR))
        col = client.get_collection(index.EXTRA_COLLECTION)
        got = col.get(where={"source": repo_name}, include=[])
        n = len(got.get("ids") or [])
        if n:
            col.delete(where={"source": repo_name})
    except Exception:
        pass
    reg = _load_registry()
    if reg.pop(repo_name, None) is not None:
        _save_registry(reg)
    return n


def load_repos_file(path: Path = REPOS_FILE) -> list[str]:
    """Список ссылок из repos.txt: по одной на строку, # — комментарии."""
    if not path.exists():
        return []
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return urls


# ──────────────────────────────── CLI ───────────────────────────────────────
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="CodeLens — добавление пользовательских репо в codelens_extra")
    ap.add_argument("url", nargs="?", help="ссылка на GitHub-репозиторий")
    ap.add_argument("--file", help="файл со списком ссылок (по умолчанию repos.txt)")
    ap.add_argument("--reindex", action="store_true",
                    help="переиндексировать, если репо уже добавлен")
    ap.add_argument("--list", action="store_true", help="показать добавленные репо")
    ap.add_argument("--remove", metavar="REPO", help="удалить репо по имени owner__repo")
    args = ap.parse_args()

    if args.list:
        repos = list_user_repos()
        if not repos:
            print("Пользовательских репозиториев нет.")
        for r in repos:
            print(f"  {r['repo_name']:32} чанков={r['n_chunks']:5}  {r['url']}")
        return

    if args.remove:
        n = remove_repo(args.remove)
        print(f"Удалено чанков: {n}  (репо {args.remove})")
        return

    urls: list[str] = []
    if args.file is not None:
        urls += load_repos_file(Path(args.file))
    elif args.url is None:                          # без url и без --file → пробуем repos.txt
        urls += load_repos_file(REPOS_FILE)
    if args.url:
        urls.append(args.url)
    if not urls:
        ap.error("нужна ссылка, либо --file, либо непустой repos.txt, либо --list/--remove")

    def prog(stage: str, detail: str = "") -> None:
        print(f"   [{stage}] {detail}".rstrip())

    ok = err = 0
    for u in urls:
        print(f"\n### {u}")
        try:
            res = index_repo(u, reindex=args.reindex, progress=prog)
        except RepoError as e:
            print(f"   ОШИБКА: {e}")
            err += 1
            continue
        if res["status"] == "duplicate":
            print(f"   уже добавлен ({res['n_chunks']} чанков) — используйте --reindex")
        else:
            print(f"   OK [{res['status']}] {res['repo_name']}: "
                  f"файлов={res.get('n_files', '?')}, чанков={res['n_chunks']}, "
                  f"fetch={res.get('fetch')}")
            ok += 1
    print(f"\nИтого: успешно {ok}, ошибок {err}.")


if __name__ == "__main__":
    _cli()

"""Чинит битые CA-переменные окружения ДО сетевых вызовов huggingface_hub.

В некоторых conda/Git-Bash-оболочках SSL_CERT_FILE (реже REQUESTS_CA_BUNDLE /
CURL_CA_BUNDLE / SSL_CERT_DIR) указывает на УДАЛЁННЫЙ cacert.pem. httpx внутри
huggingface_hub читает SSL_CERT_FILE и падает с FileNotFoundError ещё до того,
как успевает взять модель из локального кэша:

    ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]) -> FileNotFoundError

Импортируй этот модуль ПЕРВЫМ (он отрабатывает на import). Идемпотентно, без
сети: если путь битый — подменяем на bundle из certifi, иначе удаляем переменную.
Валидные значения не трогаем.
"""

import os


def fix_ca_env() -> None:
    try:
        import certifi
        cafile = certifi.where()
    except Exception:
        cafile = None

    # Переменные-файлы: битый путь -> certifi (или удалить, если certifi нет).
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var)
        if val and not os.path.isfile(val):
            if cafile:
                os.environ[var] = cafile
            else:
                os.environ.pop(var, None)

    # Переменная-директория: битый путь -> удалить (certifi сюда не подходит).
    d = os.environ.get("SSL_CERT_DIR")
    if d and not os.path.isdir(d):
        os.environ.pop("SSL_CERT_DIR", None)


fix_ca_env()

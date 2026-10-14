"""Клиент для GET-запросов к открытым API.

Минимальный, но надёжный клиент с:
  - таймаутом
  - обработкой HTTP-ошибок
  - валидацией JSON-ответа
  - безопасным хранением URL

Используется для интеграции с внешними REST-сервисами
(jsonplaceholder, open-meteo, REST Countries и др.).

Не затрагивает RAG, Pinecone, GenAPI и Telegram-бота.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Таймаут по умолчанию (секунды)
DEFAULT_TIMEOUT = 15

# Открытый API для тестирования — jsonplaceholder (публичный, без авторизации)
DEFAULT_TEST_API = "https://jsonplaceholder.typicode.com"


# ======================================================================
# Результаты запросов
# ======================================================================


@dataclass
class APIResponse:
    """Результат GET-запроса к внешнему API."""

    success: bool = False
    status_code: int = 0
    data: Any = None
    error_message: str = ""
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class APIError(Exception):
    """Ошибка при выполнении API-запроса."""

    message: str
    status_code: int = 0

    def __str__(self) -> str:
        if self.status_code:
            return f"HTTP {self.status_code}: {self.message}"
        return self.message


# ======================================================================
# Клиент
# ======================================================================


class APIClient:
    """Клиент для GET-запросов к открытым REST API.

    Примеры использования:

        client = APIClient()
        response = client.get(f"{DEFAULT_TEST_API}/posts/1")
        if response.success:
            print(response.data["title"])

        # С таймаутом и заголовками
        response = client.get(
            "https://jsonplaceholder.typicode.com/users",
            timeout=10,
            headers={"Accept": "application/json"},
        )
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        """Инициализация клиента.

        Args:
            timeout: Таймаут запроса в секундах.
        """
        self._timeout = timeout
        self._session = requests.Session()

    def get(
        self,
        url: str,
        timeout: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> APIResponse:
        """Выполняет GET-запрос к указанному URL.

        Args:
            url: Полный URL для запроса.
            timeout: Таймаут (переопределение). Если None — используется self._timeout.
            headers: Дополнительные заголовки запроса.
            params: Параметры запроса (query string).

        Returns:
            APIResponse с результатом запроса.

        Raises:
            APIError: при ошибке сети, таймауте или некорректном ответе.
        """
        timeout = timeout if timeout is not None else self._timeout

        response = APIResponse(url=url)

        # --- Выполняем запрос ---
        try:
            resp = self._session.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                allow_redirects=True,
            )
        except requests.ConnectionError as exc:
            error_msg = f"Не удалось подключиться к {url}: {exc}"
            logger.error(error_msg)
            response.error_message = error_msg
            raise APIError(message=error_msg, status_code=0) from exc
        except requests.Timeout as exc:
            error_msg = f"Превышено время ожидания ответа ({timeout} с) от {url}"
            logger.error(error_msg)
            response.error_message = error_msg
            raise APIError(message=error_msg, status_code=0) from exc
        except requests.RequestException as exc:
            error_msg = f"Ошибка запроса к {url}: {exc}"
            logger.error(error_msg)
            response.error_message = error_msg
            raise APIError(message=error_msg, status_code=0) from exc

        response.status_code = resp.status_code
        response.headers = dict(resp.headers)

        # --- Проверяем HTTP-код ---
        if resp.status_code >= 400:
            error_msg = self._extract_error_message(resp)
            logger.warning(
                "HTTP %d при запросе к %s: %s",
                resp.status_code,
                url,
                error_msg,
            )
            response.error_message = error_msg
            response.data = self._safe_json(resp)
            raise APIError(
                message=f"HTTP {resp.status_code}: {error_msg}",
                status_code=resp.status_code,
            )

        # --- Парсим JSON ---
        json_data = self._safe_json(resp)
        if json_data is None:
            error_msg = (
                f"Ответ от {url} не является валидным JSON "
                f"(HTTP {resp.status_code}, Content-Type: {resp.headers.get('content-type', 'unknown')})"
            )
            logger.error(error_msg)
            response.error_message = error_msg
            raise APIError(
                message=f"Ответ не является валидным JSON: {url}",
                status_code=resp.status_code,
            )

        response.success = True
        response.data = json_data
        logger.info("Успешный GET-запрос к %s (HTTP %d)", url, resp.status_code)
        return response

    def get_or_raise(
        self,
        url: str,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Выполняет GET-запрос и возвращает данные, или выбрасывает исключение.

        Удобно для скриптов, где нужно обработать ошибку вызовом try/except.

        Args:
            url: Полный URL.
            timeout: Таймаут.
            **kwargs: Дополнительные аргументы для get().

        Returns:
            Распарсенные данные (dict/list).

        Raises:
            APIError: при любой ошибке.
        """
        response = self.get(url, timeout=timeout, **kwargs)
        return response.data

    @staticmethod
    def _safe_json(response: requests.Response) -> Optional[Any]:
        """Безопасно парсит JSON из ответа.

        Возвращает None, если ответ пустой или не является валидным JSON.
        """
        if not response.text.strip():
            return None
        try:
            return response.json()
        except ValueError as exc:
            logger.warning(
                "Не удалось разобрать JSON из ответа (первые 200 символов): %s",
                (response.text[:200] or "").replace("\n", " "),
            )
            return None

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        """Извлекает краткое описание ошибки без секретов."""
        data = response.json() if response.headers.get("content-type", "").find("json") >= 0 else {}
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, str) and err:
            return err[:300]
        if isinstance(err, dict):
            msg = err.get("message") or err.get("detail")
            if msg:
                return str(msg)[:300]
        text = (response.text or "").strip()
        return text[:300] if text else f"HTTP {response.status_code}"

    def close(self) -> None:
        """Закрывает сессию requests."""
        self._session.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ======================================================================
# Функция-обёртка для быстрого использования
# ======================================================================


def fetch_data(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    params: Optional[dict[str, Any]] = None,
) -> APIResponse:
    """Быстрый GET-запрос к открытому API.

    Args:
        url: Полный URL.
        timeout: Таймаут в секундах.
        params: Параметры запроса.

    Returns:
        APIResponse с результатом.

    Пример:
        >>> response = fetch_data("https://jsonplaceholder.typicode.com/posts/1")
        >>> if response.success:
        ...     print(response.data["title"])
    """
    with APIClient(timeout=timeout) as client:
        return client.get(url, params=params)

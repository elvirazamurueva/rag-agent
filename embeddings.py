"""Генерация эмбеддингов через нативный API GenAPI.

ВАЖНО: эмбеддинги получаются НЕ через OpenAI-совместимый прокси
(https://proxy.gen-api.ru/v1), а через нативный API генераций GenAPI:

    POST {GENAPI_EMBEDDINGS_URL}
    https://api.gen-api.ru/api/v1/networks/embeddings

Формат запроса (нативный, отличается от OpenAI):

    {
        "model": "text-embedding-3-small",
        "input": ["текст"],
        "is_sync": true
    }

Формат ответа:

    {
        "request_id": 123,
        "model": "text-embedding-3-small",
        "cost": 0.0,
        "response": {
            "model": "text-embedding-3-small",
            "usage": {...},
            "data": [{"object": "embedding", "embedding": [ ... 1536 чисел ... ]}]
        }
    }

Ожидаемая размерность модели text-embedding-3-small — 1536.
"""

import time
from dataclasses import dataclass
from typing import Any

import requests

# Размерность модели text-embedding-3-small
EMBEDDING_DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-3-small"

# Базовый адрес нативного API GenAPI (для опроса результата в async-режиме)
_NATIVE_API_BASE = "https://api.gen-api.ru/api/v1"


@dataclass
class EmbeddingResult:
    """Результат генерации эмбеддинга."""
    values: list[float]
    dimension: int
    model: str


class EmbeddingError(Exception):
    """Ошибка при генерации эмбеддинга."""


class EmbeddingClient:
    """Клиент эмбеддингов через нативный API GenAPI."""

    def __init__(
        self,
        api_key: str,
        embeddings_url: str,
        model: str = EMBEDDING_MODEL,
        timeout: int = 30,
        poll_interval: float = 1.0,
        max_poll_attempts: int = 30,
    ) -> None:
        self._api_key = api_key
        self._url = embeddings_url
        self._model = model
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_poll_attempts = max_poll_attempts

    def create_embedding(self, text: str) -> EmbeddingResult:
        """Получает эмбеддинг для одного текста.

        Args:
            text: Входной текст.

        Returns:
            EmbeddingResult: вектор (ожидаемая размерность 1536).

        Raises:
            EmbeddingError: при ошибке запроса или неверном формате ответа.
        """
        payload = {
            "model": self._model,
            "input": [text],
            "is_sync": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.ConnectionError as exc:
            raise EmbeddingError(
                f"Не удалось подключиться к сервису эмбеддингов GenAPI: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise EmbeddingError(
                f"Превышено время ожидания ответа сервиса эмбеддингов "
                f"({self._timeout} с)."
            ) from exc

        data = self._parse_json(response, "запроса эмбеддинга")

        if response.status_code != 200:
            raise EmbeddingError(
                f"GenAPI embeddings вернул HTTP {response.status_code}: "
                f"{self._extract_error(data, response)}"
            )

        # Нативный ответ может содержать поле error даже при HTTP 200
        if data.get("error"):
            raise EmbeddingError(
                f"GenAPI embeddings сообщил об ошибке: "
                f"{self._extract_error(data, response)}"
            )

        inner = data.get("response")

        # Если результат ещё не готов (async-режим) — опрашиваем по request_id
        if not inner:
            request_id = data.get("request_id")
            if request_id is None:
                raise EmbeddingError(
                    "Неожиданный формат ответа GenAPI: отсутствует поле 'response'."
                )
            inner = self._poll_result(request_id)

        return self._extract_embedding(inner)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _poll_result(self, request_id: Any) -> dict[str, Any]:
        """Опрашивает результат генерации по request_id (async-режим)."""
        url = f"{_NATIVE_API_BASE}/request/get/{request_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        for _ in range(self._max_poll_attempts):
            time.sleep(self._poll_interval)
            try:
                response = requests.get(url, headers=headers, timeout=self._timeout)
            except requests.RequestException as exc:
                raise EmbeddingError(
                    f"Ошибка при получении результата эмбеддинга: {exc}"
                ) from exc

            data = self._parse_json(response, "получения результата")
            if response.status_code != 200:
                raise EmbeddingError(
                    f"GenAPI вернул HTTP {response.status_code} при получении "
                    f"результата: {self._extract_error(data, response)}"
                )

            inner = data.get("response")
            if inner:
                return inner

            if data.get("status") == "error":
                raise EmbeddingError(
                    f"GenAPI вернул статус 'error': "
                    f"{self._extract_error(data, response)}"
                )

        raise EmbeddingError(
            "Не удалось получить результат эмбеддинга: превышено время ожидания."
        )

    def _extract_embedding(self, inner: Any) -> EmbeddingResult:
        """Извлекает вектор из вложенного ответа GenAPI."""
        try:
            items = inner["data"]
            embedding_values = items[0]["embedding"]
            model = inner.get("model", self._model)
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                "Неожиданный формат ответа сервиса эмбеддингов: "
                "не найден вектор в поле response.data[0].embedding."
            ) from exc

        if not isinstance(embedding_values, list) or not embedding_values:
            raise EmbeddingError("Эмбеддинг не является непустым списком чисел.")

        return EmbeddingResult(
            values=embedding_values,
            dimension=len(embedding_values),
            model=model,
        )

    @staticmethod
    def _parse_json(response: requests.Response, stage: str) -> dict[str, Any]:
        """Безопасно разбирает JSON-ответ."""
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingError(
                f"Не удалось разобрать ответ GenAPI ({stage}): "
                f"получен не-JSON ответ (HTTP {response.status_code})."
            ) from exc
        if not isinstance(data, dict):
            raise EmbeddingError(
                f"Неожиданный формат ответа GenAPI ({stage}): ожидался объект JSON."
            )
        return data

    @staticmethod
    def _extract_error(data: dict[str, Any], response: requests.Response) -> str:
        """Извлекает краткое описание ошибки без секретов."""
        err = data.get("error")
        if isinstance(err, str) and err:
            return err[:300]
        if isinstance(err, dict):
            msg = err.get("message") or err.get("detail")
            if msg:
                return str(msg)[:300]
        validation = data.get("errors_validation")
        if validation:
            return str(validation)[:300]
        text = (response.text or "").strip()
        return text[:300] if text else "нет описания ошибки"

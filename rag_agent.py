"""RAG-ассистент: семантический поиск + генерация ответа через GenAPI.

Класс RAGAgent:
  1. Подключается к Pinecone и проверяет индекс.
  2. Создаёт эмбеддинг запроса через GenAPI.
  3. Ищет похожие фрагменты в векторном хранилище.
  4. Формирует ответ с помощью языковой модели.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI
from pinecone import Pinecone

from embeddings import EmbeddingClient, EmbeddingError, EMBEDDING_DIMENSION
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Один найденный фрагмент из векторного хранилища."""
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text_preview: str = ""


@dataclass
class SearchResults:
    """Результат поиска: список фрагментов или статус."""
    found: bool
    results: list[SearchResult] = field(default_factory=list)
    total_vectors: int = 0
    message: str = ""


@dataclass
class RAGResponse:
    """Полный ответ RAG-ассистента."""
    query: str
    search: SearchResults
    answer: str = ""
    has_answer: bool = False
    error: str = ""


class PineconeError(Exception):
    """Ошибка при работе с Pinecone."""


class RAGEngine:
    """Ядро RAG: поиск в Pinecone + генерация ответа через GenAPI."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._embedding_client = EmbeddingClient(
            api_key=config.genapi_api_key,
            embeddings_url=config.genapi_embeddings_url,
            model=config.embedding_model,
            timeout=config.timeout_genapi,
        )
        self._pc: Optional[Pinecone] = None
        self._index: Optional[Any] = None

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def connect_pinecone(self) -> None:
        """Подключается к Pinecone и получает handle индекса."""
        try:
            kwargs: dict[str, Any] = {"api_key": self._config.pinecone_api_key}
            proxy = self._config.pinecone_proxy_url
            if proxy:
                # httpx требует схему (http:// или https://).
                # Если пользователь указал только user:pass@host:port — добавляем.
                if not proxy.lower().startswith(("http://", "https://")):
                    proxy = "http://" + proxy
                kwargs["proxy_url"] = proxy
            self._pc = Pinecone(**kwargs)
            logger.info("Клиент Pinecone инициализирован.")
        except Exception as exc:
            raise PineconeError(f"Ошибка инициализации Pinecone: {exc}") from exc

        try:
            self._index = self._pc.index(name=self._config.pinecone_index_name)
            logger.info("Индекс '%s' получен.", self._config.pinecone_index_name)
        except Exception as exc:
            raise PineconeError(
                f"Ошибка получения индекса '{self._config.pinecone_index_name}': {exc}"
            ) from exc

    def check_index(self) -> dict[str, Any]:
        """Проверяет индекс: существование, размерность, количество векторов.

        Returns:
            dict с ключами: exists (bool), dimension (int | None),
            total_vectors (int), info (str).
        """
        info = {
            "exists": False,
            "dimension": None,
            "total_vectors": 0,
            "info": "",
        }
        try:
            stats = self._index.describe_index_stats()
            info["exists"] = True
            info["dimension"] = getattr(stats, "dimension", None)
            info["total_vectors"] = getattr(stats, "total_vector_count", 0)
            info["info"] = (
                f"Индекс существует. Размерность: {info['dimension']}. "
                f"Векторов: {info['total_vectors']}."
            )
            logger.info(info["info"])
        except Exception as exc:
            info["info"] = f"Ошибка проверки индекса: {exc}"
            logger.error(info["info"])
        return info

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------

    def create_embedding(self, text: str) -> list[float]:
        """Создаёт эмбеддинг для текста.

        Returns:
            Список float-значений.

        Raises:
            EmbeddingError: при ошибке сервиса эмбеддингов.
        """
        result = self._embedding_client.create_embedding(text)
        if result.dimension != EMBEDDING_DIMENSION:
            logger.warning(
                "Размерность эмбеддинга %d не совпадает с ожидаемой %d",
                result.dimension,
                EMBEDDING_DIMENSION,
            )
        return result.values

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResults:
        """Ищет похожие фрагменты в Pinecone.

        Args:
            query: Текст запроса.
            top_k: Количество результатов.

        Returns:
            SearchResults с найденными фрагментами или сообщением.
        """
        # Создаём эмбеддинг
        try:
            embedding = self.create_embedding(query)
        except EmbeddingError as exc:
            return SearchResults(
                found=False,
                message=f"Ошибка генерации эмбеддинга: {exc}",
            )

        # Выполняем поиск
        try:
            response = self._index.query(
                top_k=top_k,
                vector=embedding,
                include_metadata=True,
            )
        except Exception as exc:
            return SearchResults(
                found=False,
                message=f"Ошибка запроса к Pinecone: {exc}",
            )

        matches = response.matches
        if not matches:
            return SearchResults(
                found=False,
                message="Совпадений не найдено. Индекс может быть пустым или "
                        "запрос не имеет семантического совпадения.",
            )

        results = []
        for match in matches:
            meta = dict(match.metadata) if match.metadata else {}
            # Извлекаем текст из метаданных
            text_preview = ""
            for key in ("text", "content", "chunk", "summary"):
                if key in meta:
                    text_preview = str(meta[key])[:300]
                    break

            results.append(SearchResult(
                id=match.id,
                score=match.score,
                metadata=meta,
                text_preview=text_preview,
            ))

        return SearchResults(
            found=True,
            results=results,
            total_vectors=getattr(self._index.describe_index_stats(), "total_vector_count", 0),
        )

    # ------------------------------------------------------------------
    # Генерация ответа
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, context: str) -> str:
        """Генерирует ответ на основе контекста через GenAPI.

        Args:
            query: Исходный вопрос пользователя.
            context: Контекст из найденных фрагментов.

        Returns:
            Ответ языковой модели.
        """
        client = OpenAI(
            api_key=self._config.genapi_api_key,
            base_url=self._extract_base_url(self._config.genapi_url),
            timeout=self._config.timeout_genapi,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Вы — полезный ассистент. Отвечайте на вопросы, "
                    "используя только предоставленный контекст. "
                    "Если в контексте нет ответа, скажите об этом "
                    "прямо, не выдумывая информацию."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {query}\n\n"
                    f"Контекст:\n{context}"
                ),
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=self._config.model_id,
                messages=messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            return f"Ошибка генерации ответа: {exc}"

    @staticmethod
    def _extract_base_url(full_url: str) -> str:
        """Извлекает base URL из полного URL endpoint.

        Пример:
            https://proxy.gen-api.ru/v1/chat/completions
            → https://proxy.gen-api.ru/v1
        """
        if full_url.endswith("/chat/completions"):
            return full_url[: -len("/chat/completions")]
        if full_url.endswith("/embeddings"):
            return full_url[: -len("/embeddings")]
        return full_url.rstrip("/")

    # ------------------------------------------------------------------
    # Основной метод: вопрос → ответ
    # ------------------------------------------------------------------

    def ask(self, query: str, top_k: int = 5) -> RAGResponse:
        """Выполняет полный цикл RAG: поиск + генерация ответа.

        Args:
            query: Вопрос пользователя.
            top_k: Количество результатов поиска.

        Returns:
            RAGResponse с результатами поиска и ответом.
        """
        search = self.search(query, top_k=top_k)

        if not search.found:
            return RAGResponse(
                query=query,
                search=search,
                has_answer=False,
            )

        # Формируем контекст из найденных фрагментов
        parts: list[str] = []
        for i, r in enumerate(search.results, 1):
            if r.text_preview:
                parts.append(f"[{i}] {r.text_preview}")
            else:
                parts.append(f"[{i}] (нет текста, ID: {r.id})")
        context = "\n\n".join(parts)

        answer = self.generate_answer(query, context)
        has_answer = (
            answer and "не содержит" not in answer.lower()
            and "нет информации" not in answer.lower()
            and "не хватает" not in answer.lower()
            and "недостаточно" not in answer.lower()
            and "не могу ответить" not in answer.lower()
        )

        return RAGResponse(
            query=query,
            search=search,
            answer=answer,
            has_answer=has_answer,
        )

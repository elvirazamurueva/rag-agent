"""Автоматические тесты для RAG-ассистента.

Тесты не требуют реальных API-ключей и не изменяют настоящую базу Pinecone.
Используются имитации внешних сервисов.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, load_config
from embeddings import EmbeddingClient, EmbeddingResult, EmbeddingError, EMBEDDING_DIMENSION
from rag_agent import RAGEngine, SearchResult, SearchResults, RAGResponse


# ======================================================================
# Тесты конфигурации
# ======================================================================

class TestConfig:
    """Проверка загрузки и валидации конфигурации."""

    def test_config_creation_with_valid_values(self):
        """Создание конфигурации с корректными значениями."""
        cfg = Config(
            genapi_api_key="test-key",
            pinecone_api_key="pc-test-key",
            pinecone_index_name="test-index",
        )
        assert cfg.genapi_api_key == "test-key"
        assert cfg.pinecone_index_name == "test-index"
        assert cfg.model_id == "gpt-4o-mini"
        assert cfg.genapi_url == "https://proxy.gen-api.ru/v1/chat/completions"

    def test_config_missing_genapi_key(self):
        """Ошибка при отсутствии ключа GenAPI."""
        with pytest.raises(ValueError, match="GENAPI_API_KEY"):
            Config(
                genapi_api_key="",
                pinecone_api_key="pc-test-key",
                pinecone_index_name="test-index",
            )

    def test_config_missing_pinecone_key(self):
        """Ошибка при отсутствии ключа Pinecone."""
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            Config(
                genapi_api_key="test-key",
                pinecone_api_key="",
                pinecone_index_name="test-index",
            )

    def test_config_missing_index_name(self):
        """Ошибка при отсутствии имени индекса."""
        with pytest.raises(ValueError, match="PINECONE_INDEX_NAME"):
            Config(
                genapi_api_key="test-key",
                pinecone_api_key="pc-test-key",
                pinecone_index_name="",
            )

    def test_config_all_missing(self):
        """Сообщение содержит все отсутствующие переменные."""
        with pytest.raises(ValueError) as exc_info:
            Config(
                genapi_api_key="",
                pinecone_api_key="",
                pinecone_index_name="",
            )
        msg = str(exc_info.value)
        assert "GENAPI_API_KEY" in msg
        assert "PINECONE_API_KEY" in msg
        assert "PINECONE_INDEX_NAME" in msg

    def test_config_no_secrets_in_error_message(self):
        """В сообщении об ошибке нет значений ключей."""
        with pytest.raises(ValueError) as exc_info:
            Config(
                genapi_api_key="",
                pinecone_api_key="",
                pinecone_index_name="",
            )
        # Убеждаемся, что в сообщении нет длинных строк, похожих на ключи
        msg = str(exc_info.value)
        assert "sk-" not in msg
        assert "pc-" not in msg


# ======================================================================
# Тесты эмбеддингов
# ======================================================================

class TestEmbeddings:
    """Проверка клиента эмбеддингов с имитацией HTTP (нативный API GenAPI)."""

    def _make_client(self):
        return EmbeddingClient(
            api_key="test-key",
            embeddings_url="https://api.gen-api.ru/api/v1/networks/embeddings",
            model="text-embedding-3-small",
            timeout=10,
        )

    @staticmethod
    def _native_response(fake_values):
        """Имитация нативного ответа GenAPI с вложенным полем response."""
        return {
            "request_id": 123,
            "organization_id": 1,
            "model": "text-embedding-3-small",
            "cost": 0.0,
            "response": {
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "data": [{"object": "embedding", "embedding": fake_values}],
            },
        }

    def test_create_embedding_success(self):
        """Успешное получение эмбеддинга (нативный формат)."""
        client = self._make_client()
        fake_values = [0.1] * EMBEDDING_DIMENSION

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = self._native_response(fake_values)

        with patch("embeddings.requests.post", return_value=mock_response) as mock_post:
            result = client.create_embedding("тестовый текст")

        assert result.dimension == EMBEDDING_DIMENSION
        assert len(result.values) == EMBEDDING_DIMENSION
        assert result.model == "text-embedding-3-small"
        mock_post.assert_called_once()

        # Проверяем, что отправлен нативный формат (input как список, is_sync)
        _, kwargs = mock_post.call_args
        sent = kwargs["json"]
        assert sent["input"] == ["тестовый текст"]
        assert sent["is_sync"] is True
        assert sent["model"] == "text-embedding-3-small"

    def test_create_embedding_http_error(self):
        """Ошибка HTTP при получении эмбеддинга."""
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = ""
        mock_response.json.return_value = {"error": "Invalid API key"}

        with patch("embeddings.requests.post", return_value=mock_response):
            with pytest.raises(EmbeddingError, match="HTTP 401"):
                client.create_embedding("тест")

    def test_create_embedding_api_error_in_body(self):
        """Ошибка в теле ответа при HTTP 200."""
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"error": "model not found"}

        with patch("embeddings.requests.post", return_value=mock_response):
            with pytest.raises(EmbeddingError, match="model not found"):
                client.create_embedding("тест")

    def test_create_embedding_connection_error(self):
        """Ошибка соединения при получении эмбеддинга."""
        import requests as req

        client = self._make_client()

        with patch(
            "embeddings.requests.post",
            side_effect=req.ConnectionError("Connection refused"),
        ):
            with pytest.raises(EmbeddingError, match="подключиться"):
                client.create_embedding("тест")

    def test_create_embedding_timeout(self):
        """Тайм-аут при получении эмбеддинга."""
        import requests as req

        client = self._make_client()

        with patch(
            "embeddings.requests.post",
            side_effect=req.Timeout("Timeout"),
        ):
            with pytest.raises(EmbeddingError, match="время ожидания"):
                client.create_embedding("тест")

    def test_create_embedding_unexpected_format(self):
        """Неожиданный формат ответа."""
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"unexpected": "data"}

        with patch("embeddings.requests.post", return_value=mock_response):
            with pytest.raises(EmbeddingError, match="response"):
                client.create_embedding("тест")

    def test_create_embedding_non_json(self):
        """Ответ не является JSON."""
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>error</html>"
        mock_response.json.side_effect = ValueError("not json")

        with patch("embeddings.requests.post", return_value=mock_response):
            with pytest.raises(EmbeddingError, match="не-JSON"):
                client.create_embedding("тест")


# ======================================================================
# Тесты RAGEngine (с имитацией Pinecone и GenAPI)
# ======================================================================

def _make_config() -> Config:
    """Создаёт тестовую конфигурацию."""
    return Config(
        genapi_api_key="test-genapi-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
    )


def _make_engine() -> RAGEngine:
    """Создаёт RAGEngine с имитацией внешних сервисов."""
    config = _make_config()
    engine = RAGEngine(config)
    # Подключаем фейковый Pinecone
    engine._pc = MagicMock()
    engine._index = MagicMock()
    return engine


class TestRAGEngine:
    """Проверка логики RAGEngine."""

    def test_search_found_results(self):
        """Поиск возвращает найденные результаты."""
        engine = _make_engine()

        # Имитируем эмбеддинг
        fake_embedding = [0.1] * EMBEDDING_DIMENSION
        with patch.object(
            engine._embedding_client,
            "create_embedding",
            return_value=EmbeddingResult(
                values=fake_embedding,
                dimension=EMBEDDING_DIMENSION,
                model="text-embedding-3-small",
            ),
        ):
            # Имитируем ответ Pinecone
            match1 = MagicMock()
            match1.id = "doc-1"
            match1.score = 0.95
            match1.metadata = {"text": "Векторная база данных — это..."}

            mock_response = MagicMock()
            mock_response.matches = [match1]
            engine._index.query.return_value = mock_response

            # Имитируем describe_index_stats для total_vectors
            mock_stats = MagicMock()
            mock_stats.total_vector_count = 100
            engine._index.describe_index_stats.return_value = mock_stats

            results = engine.search("Что такое векторная БД?")

        assert results.found is True
        assert len(results.results) == 1
        assert results.results[0].id == "doc-1"
        assert results.results[0].score == 0.95

    def test_search_no_matches(self):
        """Поиск не находит совпадений."""
        engine = _make_engine()

        fake_embedding = [0.1] * EMBEDDING_DIMENSION
        with patch.object(
            engine._embedding_client,
            "create_embedding",
            return_value=EmbeddingResult(
                values=fake_embedding,
                dimension=EMBEDDING_DIMENSION,
                model="text-embedding-3-small",
            ),
        ):
            mock_response = MagicMock()
            mock_response.matches = []
            engine._index.query.return_value = mock_response

            results = engine.search("Уникальный вопрос")

        assert results.found is False
        assert "не найдено" in results.message.lower() or "пуст" in results.message.lower()

    def test_search_embedding_error(self):
        """Поиск при ошибке генерации эмбеддинга."""
        engine = _make_engine()

        with patch.object(
            engine._embedding_client,
            "create_embedding",
            side_effect=EmbeddingError("Сервис недоступен"),
        ):
            results = engine.search("Что такое БД?")

        assert results.found is False
        assert "эмбеддинг" in results.message.lower()

    def test_search_pinecone_error(self):
        """Поиск при ошибке Pinecone."""
        engine = _make_engine()

        fake_embedding = [0.1] * EMBEDDING_DIMENSION
        with patch.object(
            engine._embedding_client,
            "create_embedding",
            return_value=EmbeddingResult(
                values=fake_embedding,
                dimension=EMBEDDING_DIMENSION,
                model="text-embedding-3-small",
            ),
        ):
            engine._index.query.side_effect = Exception("Pinecone connection lost")
            results = engine.search("Что такое БД?")

        assert results.found is False
        assert "pinecone" in results.message.lower()

    def test_check_index_empty(self):
        """Проверка пустого индекса."""
        engine = _make_engine()

        mock_stats = MagicMock()
        mock_stats.dimension = 1536
        mock_stats.total_vector_count = 0
        engine._index.describe_index_stats.return_value = mock_stats

        info = engine.check_index()

        assert info["exists"] is True
        assert info["dimension"] == 1536
        assert info["total_vectors"] == 0

    def test_check_index_error(self):
        """Ошибка при проверке индекса."""
        engine = _make_engine()
        engine._index.describe_index_stats.side_effect = Exception("Index not found")

        info = engine.check_index()

        assert info["exists"] is False
        assert "ошибка" in info["info"].lower()

    def test_extract_base_url_chat(self):
        """Извлечение base URL из endpoint чата."""
        assert (
            RAGEngine._extract_base_url("https://proxy.gen-api.ru/v1/chat/completions")
            == "https://proxy.gen-api.ru/v1"
        )

    def test_extract_base_url_embeddings(self):
        """Извлечение base URL из endpoint эмбеддингов."""
        assert (
            RAGEngine._extract_base_url("https://proxy.gen-api.ru/v1/embeddings")
            == "https://proxy.gen-api.ru/v1"
        )

    def test_connect_pinecone_with_proxy_no_scheme(self):
        """connect_pinecone добавляет http://, если схемы нет."""
        config = Config(
            genapi_api_key="test-genapi-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            pinecone_proxy_url="user369007:enw6dy@140.99.236.149:9325",
        )
        engine = RAGEngine(config)

        with patch("rag_agent.Pinecone") as mock_pinecone:
            engine.connect_pinecone()
            call_kwargs = mock_pinecone.call_args.kwargs
            # Должно добавиться http://
            assert call_kwargs["proxy_url"] == "http://user369007:enw6dy@140.99.236.149:9325"

    def test_connect_pinecone_with_proxy_has_scheme(self):
        """connect_pinecone не трогает URL, если схема уже есть."""
        config = Config(
            genapi_api_key="test-genapi-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            pinecone_proxy_url="http://user:pass@10.0.0.1:3128",
        )
        engine = RAGEngine(config)

        with patch("rag_agent.Pinecone") as mock_pinecone:
            engine.connect_pinecone()
            call_kwargs = mock_pinecone.call_args.kwargs
            # Без изменений
            assert call_kwargs["proxy_url"] == "http://user:pass@10.0.0.1:3128"

    def test_connect_pinecone_no_proxy(self):
        """connect_pinecone без прокси — proxy_url не передаётся."""
        config = Config(
            genapi_api_key="test-genapi-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
        )
        engine = RAGEngine(config)

        with patch("rag_agent.Pinecone") as mock_pinecone:
            engine.connect_pinecone()
            call_kwargs = mock_pinecone.call_args.kwargs
            assert "proxy_url" not in call_kwargs

    def test_ask_empty_index(self):
        """Полный цикл RAG при пустом индексе."""
        engine = _make_engine()

        fake_embedding = [0.1] * EMBEDDING_DIMENSION
        with patch.object(
            engine._embedding_client,
            "create_embedding",
            return_value=EmbeddingResult(
                values=fake_embedding,
                dimension=EMBEDDING_DIMENSION,
                model="text-embedding-3-small",
            ),
        ):
            mock_response = MagicMock()
            mock_response.matches = []
            engine._index.query.return_value = mock_response

            result = engine.ask("Что такое БД?")

        assert result.has_answer is False
        assert result.search.found is False

    def test_no_secrets_in_search_results(self):
        """В результатах поиска нет секретов."""
        engine = _make_engine()

        fake_embedding = [0.1] * EMBEDDING_DIMENSION
        with patch.object(
            engine._embedding_client,
            "create_embedding",
            return_value=EmbeddingResult(
                values=fake_embedding,
                dimension=EMBEDDING_DIMENSION,
                model="text-embedding-3-small",
            ),
        ):
            match1 = MagicMock()
            match1.id = "doc-1"
            match1.score = 0.95
            match1.metadata = {"text": "Обычный текст без секретов"}

            mock_response = MagicMock()
            mock_response.matches = [match1]
            engine._index.query.return_value = mock_response

            mock_stats = MagicMock()
            mock_stats.total_vector_count = 10
            engine._index.describe_index_stats.return_value = mock_stats

            results = engine.search("Тестовый запрос")

        # Проверяем, что ключи API не попали в результаты
        for r in results.results:
            assert "test-genapi-key" not in str(r.metadata)
            assert "test-pinecone-key" not in str(r.metadata)


# ======================================================================
# Тесты данных (dataclass)
# ======================================================================

class TestDataclasses:
    """Проверка структур данных."""

    def test_search_result_defaults(self):
        r = SearchResult(id="doc-1", score=0.5)
        assert r.metadata == {}
        assert r.text_preview == ""

    def test_search_results_not_found(self):
        r = SearchResults(found=False, message="Пусто")
        assert r.results == []
        assert r.total_vectors == 0

    def test_rag_response_defaults(self):
        sr = SearchResults(found=False)
        r = RAGResponse(query="вопрос", search=sr)
        assert r.answer == ""
        assert r.has_answer is False

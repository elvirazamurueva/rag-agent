"""Автоматические тесты для Telegram-бота RAG-ассистента.

Тесты используют моки и не требуют реальных API-ключей.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from telegram_bot import (
    _start,
    _add_url,
    _search,
    _handle_message,
    _truncate_message,
    _error_handler,
    _do_add_url,
    _do_ask,
    create_bot_application,
    MAX_TELEGRAM_MESSAGE_LENGTH,
)
from config import Config
from rag_agent import PineconeError, RAGResponse


# ======================================================================
# Вспомогательные функции для создания моков
# ======================================================================

def _make_config(
    has_token: bool = True,
    has_proxy: bool = False,
) -> Config:
    """Создаёт тестовую конфигурацию."""
    return Config(
        genapi_api_key="test-genapi-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        telegram_bot_token="123456:ABCdefGHIjklMNOpqrsTUVwxyz" if has_token else None,
        proxy_url="user:pass@proxy:8080" if has_proxy else None,
    )


def _make_context() -> MagicMock:
    """Создаёт моковый ContextTypes."""
    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.send_message = AsyncMock()
    context.bot.edit_message_text = AsyncMock()
    context.args = []
    return context


def _make_update_mock(
    text: str = "/start",
    chat_id: int = 123456,
) -> MagicMock:
    """Создаёт моковый объект Update."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = text
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_message = update.message
    return update


# ======================================================================
# Тесты вспомогательных функций
# ======================================================================

class TestHelperFunctions:
    """Тесты вспомогательных функций."""

    def test_truncate_message_short(self):
        """Короткое сообщение не обрезается."""
        text = "Короткий текст"
        assert _truncate_message(text) == text

    def test_truncate_message_long(self):
        """Длинное сообщение обрезается."""
        text = "x" * 5000
        result = _truncate_message(text)
        assert len(result) <= MAX_TELEGRAM_MESSAGE_LENGTH
        assert result.endswith("...")

    def test_truncate_message_exact_length(self):
        """Сообщение ровно максимальной длины не обрезается."""
        text = "x" * MAX_TELEGRAM_MESSAGE_LENGTH
        assert _truncate_message(text) == text


# ======================================================================
# Тесты асинхронных обработчиков команд
# ======================================================================

class TestAsyncHandlers:
    """Тесты асинхронных обработчиков."""

    def _run_async(self, coro):
        """Помощник для запуска async-функции в тесте."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_start_sends_help_message(self):
        """Команда /start отправляет сообщение с помощью."""
        update = _make_update_mock(text="/start")
        context = _make_context()

        await _start(update, context)

        context.bot.send_message.assert_called_once()
        call_args = context.bot.send_message.call_args
        assert call_args.kwargs["chat_id"] == 123456
        assert "Привет!" in call_args.kwargs["text"]
        assert "/add_url" in call_args.kwargs["text"]
        assert "/search" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_add_url_no_args(self):
        """Команда /add_url без аргументов показывает использование."""
        update = _make_update_mock(text="/add_url")
        context = _make_context()

        await _add_url(update, context)

        context.bot.send_message.assert_called_once()
        assert "Использование" in context.bot.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_search_no_args(self):
        """Команда /search без аргументов показывает использование."""
        update = _make_update_mock(text="/search")
        context = _make_context()

        await _search(update, context)

        context.bot.send_message.assert_called_once()
        assert "Использование" in context.bot.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_handle_message_success(self):
        """Обработка обычного вопроса."""
        update = _make_update_mock(text="Вопрос без команды")
        context = _make_context()

        with patch("telegram_bot._do_ask") as mock_ask:
            mock_ask.return_value = "Ответ на вопрос."
            await _handle_message(update, context)

            context.bot.send_message.assert_called_once()
            context.bot.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_empty_text(self):
        """Пустой текст не обрабатывается."""
        update = _make_update_mock(text="")
        update.message.text = "   "
        context = _make_context()

        await _handle_message(update, context)

        # Не должно быть вызовов API
        context.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handler_sends_message(self):
        """Обработчик ошибок отправляет сообщение пользователю."""
        update = _make_update_mock(text="/search Вопрос")
        context = MagicMock()
        context.bot = AsyncMock()
        context.bot.send_message = AsyncMock()
        context.error = Exception("Test error")

        await _error_handler(update, context)

        context.bot.send_message.assert_called_once()
        assert "ошибк" in context.bot.send_message.call_args.kwargs["text"].lower()


# ======================================================================
# Тесты синхронных RAG-функций (моки)
# ======================================================================

class TestSyncRAGFunctions:
    """Тесты синхронных функций _do_add_url и _do_ask."""

    @patch("telegram_bot.load_config")
    @patch("telegram_bot.RAGEngine")
    @patch("telegram_bot.DocumentLoader")
    def test_do_add_url_success(self, MockLoader, MockEngine, mock_load_config):
        """Успешная загрузка URL через _do_add_url."""
        mock_load_config.return_value = _make_config()
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance

        mock_doc = MagicMock()
        mock_doc.success = True
        mock_doc.source_url = "http://example.com"
        mock_doc.page_title = "Test Page"
        mock_doc.full_text = "Test content"
        mock_doc.total_chunks = 1
        mock_doc.saved_vectors = 1
        MockLoader.return_value.load_url.return_value = mock_doc

        result = _do_add_url("http://example.com")

        assert "успешно загружен" in result
        assert "Test Page" in result

    @patch("telegram_bot.load_config")
    @patch("telegram_bot.RAGEngine")
    @patch("telegram_bot.DocumentLoader")
    def test_do_add_url_error(self, MockLoader, MockEngine, mock_load_config):
        """Ошибка загрузки URL через _do_add_url."""
        mock_load_config.return_value = _make_config()
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance

        mock_doc = MagicMock()
        mock_doc.success = False
        mock_doc.error_message = "Некорректный URL"
        MockLoader.return_value.load_url.return_value = mock_doc

        result = _do_add_url("http://invalid")

        assert "Ошибка" in result
        assert "Некорректный URL" in result

    @patch("telegram_bot.load_config")
    @patch("telegram_bot.RAGEngine")
    def test_do_ask_success(self, MockEngine, mock_load_config):
        """Успешный поиск через _do_ask."""
        mock_load_config.return_value = _make_config()
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance

        mock_search = MagicMock()
        mock_search.found = True
        mock_search.results = [
            MagicMock(metadata={"source_url": "http://example.com"}, text_preview="Тест")
        ]
        mock_search.message = ""
        mock_answer = RAGResponse(
            query="test",
            search=mock_search,
            answer="Это тестовый ответ.",
            has_answer=True,
        )
        engine_instance.ask.return_value = mock_answer

        result = _do_ask("Тестовый вопрос")

        assert "тестовый ответ" in result.lower()

    @patch("telegram_bot.load_config")
    @patch("telegram_bot.RAGEngine")
    def test_do_ask_no_results(self, MockEngine, mock_load_config):
        """Поиск без результатов через _do_ask."""
        mock_load_config.return_value = _make_config()
        engine_instance = MagicMock()
        MockEngine.return_value = engine_instance

        mock_search = MagicMock()
        mock_search.found = False
        mock_search.message = "Совпадений не найдено"
        mock_answer = RAGResponse(
            query="test",
            search=mock_search,
            answer="",
            has_answer=False,
        )
        engine_instance.ask.return_value = mock_answer

        result = _do_ask("Несуществующий вопрос")

        assert "не найдено" in result.lower()


# ======================================================================
# Тесты создания приложения
# ======================================================================

class TestBotApplication:
    """Тесты создания приложения бота."""

    def test_create_bot_application_no_proxy(self):
        """Создание приложения без прокси."""
        config = _make_config(has_proxy=False)

        with patch(
            "telegram_bot.Application.builder"
        ) as MockBuilder:
            mock_builder_instance = MagicMock()
            MockBuilder.return_value = mock_builder_instance
            mock_app = MagicMock()
            mock_builder_instance.build.return_value = mock_app

            application = create_bot_application(config)

            # Проверяем, что указан токен
            mock_builder_instance.token.assert_called_once_with(
                config.telegram_bot_token
            )
            # Проверяем, что proxy не вызывается
            mock_builder_instance.proxy.assert_not_called()

    @patch("telegram_bot.Application.builder")
    def test_create_bot_application_with_proxy(self, MockBuilder):
        """Создание приложения с прокси (HTTPXRequest с trust_env=False)."""
        mock_builder_instance = MagicMock()
        MockBuilder.return_value = mock_builder_instance
        mock_app = MagicMock()
        mock_builder_instance.build.return_value = mock_app

        application = create_bot_application(
            _make_config(has_proxy=True)
        )

        # Проверяем, что указан токен
        mock_builder_instance.token.assert_called_once()
        # Проверяем, что создан HTTPXRequest с proxy и trust_env=False
        mock_builder_instance.request.assert_called_once()
        mock_builder_instance.get_updates_request.assert_called_once()

        # Проверяем, что request — это HTTPXRequest с trust_env=False
        request_call = mock_builder_instance.request.call_args
        # Создадим реальную HTTPXRequest с теми же параметрами для проверки
        from telegram.request import HTTPXRequest
        real_request = HTTPXRequest(
            proxy="http://user:pass@proxy:8080",
            httpx_kwargs={"trust_env": False},
        )
        assert real_request._client_kwargs["trust_env"] is False

    def test_create_bot_application_adds_handlers(self):
        """Проверка регистрации обработчиков команд."""
        config = _make_config()

        with patch(
            "telegram_bot.Application.builder"
        ) as MockBuilder:
            mock_builder_instance = MagicMock()
            MockBuilder.return_value = mock_builder_instance
            mock_app = MagicMock()
            mock_builder_instance.build.return_value = mock_app

            application = create_bot_application(config)

            # Проверяем, что добавлены обработчики
            assert mock_app.add_handler.call_count >= 3  # start, add_url, search


# ======================================================================
# Тесты конфигурации
# ======================================================================

class TestConfigIntegration:
    """Тесты интеграции с конфигурацией."""

    def test_config_with_telegram_fields(self):
        """Конфигурация содержит поля для Telegram."""
        config = _make_config(has_token=True, has_proxy=True)

        assert config.telegram_bot_token is not None
        assert config.proxy_url is not None

    def test_config_without_telegram_fields(self):
        """Конфигурация без полей Telegram."""
        config = _make_config(has_token=False, has_proxy=False)

        assert config.telegram_bot_token is None
        assert config.proxy_url is None


# ======================================================================
# Запуск тестов
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Telegram-бот для RAG-ассистента.

Поддерживает команды:
  /start — объяснение возможностей бота
  /add_url <URL> — загрузка веб-страницы в Pinecone
  /search <вопрос> — вопрос по загруженным документам

Бот использует существующие RAGEngine и DocumentLoader.

ВАЖНО: python-telegram-bot v20+ требует async-обработчики.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config
from rag_agent import RAGEngine, PineconeError
from document_loader import DocumentLoader, DocumentLoaderError, SSRFError

logger = logging.getLogger(__name__)

# Максимальная длина сообщения в Telegram (с учётом лимита платформы)
MAX_TELEGRAM_MESSAGE_LENGTH = 4096


def _truncate_message(message: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> str:
    """Обрезает сообщение до максимальной длины Telegram."""
    if len(message) <= max_length:
        return message
    return message[: max_length - 3] + "..."


def _build_engine_and_loader(config: Config):
    """Создаёт RAGEngine и DocumentLoader (синхронная часть)."""
    engine = RAGEngine(config)
    engine.connect_pinecone()
    loader = DocumentLoader(
        embedding_client=engine._embedding_client,
        index=engine._index,
        chunk_size=500,
        chunk_overlap=50,
    )
    return engine, loader


def _do_add_url(url: str) -> str:
    """Синхронная загрузка URL (вызывается в отдельном потоке)."""
    config = load_config()
    engine, loader = _build_engine_and_loader(config)
    doc = loader.load_url(url)

    if doc.success:
        return (
            f"✅ Документ успешно загружен!\n\n"
            f"📌 Источник: {doc.source_url}\n"
            f"📄 Заголовок: {doc.page_title or 'Не указан'}\n"
            f"📝 Символов: {len(doc.full_text)}\n"
            f"🔢 Фрагментов: {doc.total_chunks}\n"
            f"📊 Векторов сохранено: {doc.saved_vectors}"
        )
    else:
        return f"❌ Ошибка загрузки: {doc.error_message}"


def _do_ask(question: str) -> str:
    """Синхронный RAG-запрос (вызывается в отдельном потоке)."""
    config = load_config()
    engine = RAGEngine(config)
    engine.connect_pinecone()
    result = engine.ask(question, top_k=5)

    answer_parts = []

    if result.search.found and result.search.results:
        sources = set()
        for r in result.search.results:
            url = r.metadata.get("source_url", "")
            if url:
                sources.add(url)

        answer_parts.append(result.answer or "Ответ не сгенерирован.")

        if sources:
            sources_text = "\n".join(f"• {s}" for s in sorted(sources))
            answer_parts.append(f"\n\n📚 Источники:\n{sources_text}")
    else:
        answer_parts.append(
            result.search.message
            or "Не удалось найти релевантные документы."
        )

    return _truncate_message("\n".join(answer_parts))


# ======================================================================
# Async-обработчики команд
# ======================================================================

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок бота."""
    logger.error("Исключение в боте:", exc_info=context.error)
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Произошла ошибка при обработке сообщения. Попробуйте позже.",
        )


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    message = (
        "Привет! Я RAG-ассистент. Я могу:\n\n"
        "📄 /add_url <URL> — загрузить веб-страницу для поиска по ней\n"
        "❓ /search <вопрос> — задать вопрос по загруженным документам\n"
        "💬 Или просто отправьте сообщение с вопросом — я постараюсь ответить.\n\n"
        "Примеры:\n"
        "• /add_url https://example.com/article\n"
        "• /search Что такое искусственный интеллект?"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)


async def _add_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /add_url."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Использование: /add_url <URL>\n\nПример:\n/add_url https://example.com/article",
        )
        return

    url = " ".join(context.args)

    # Отправляем сообщение о начале загрузки
    loading_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="⏳ Загрузка и обработка страницы...",
    )

    try:
        # Тяжёлую работу выносим в отдельный поток
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(None, _do_add_url, url)

        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=message,
        )

    except DocumentLoaderError as exc:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=f"❌ Ошибка: {exc}",
        )
    except Exception as exc:
        logger.error("Ошибка в /add_url:", exc_info=exc)
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text="❌ Произошла ошибка при загрузке документа. Попробуйте позже.",
        )


async def _search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /search."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Использование: /search <вопрос>\n\nПример:\n/search Что такое RAG?",
        )
        return

    question = " ".join(context.args)

    # Отправляем сообщение о начале обработки
    loading_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="⏳ Ищу ответ...",
    )

    try:
        loop = asyncio.get_event_loop()
        full_answer = await loop.run_in_executor(None, _do_ask, question)

        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=full_answer,
        )

    except PineconeError as exc:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=f"❌ Ошибка Pinecone: {exc}",
        )
    except Exception as exc:
        logger.error("Ошибка в /search:", exc_info=exc)
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text="❌ Произошла ошибка при поиске ответа. Попробуйте позже.",
        )


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик обычных сообщений (вопросов)."""
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()

    if not question:
        return

    # Отправляем сообщение о начале обработки
    loading_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="⏳ Ищу ответ...",
    )

    try:
        loop = asyncio.get_event_loop()
        full_answer = await loop.run_in_executor(None, _do_ask, question)

        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=full_answer,
        )

    except PineconeError as exc:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text=f"❌ Ошибка Pinecone: {exc}",
        )
    except Exception as exc:
        logger.error("Ошибка в обработке сообщения:", exc_info=exc)
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=loading_msg.message_id,
            text="❌ Произошла ошибка при поиске ответа. Попробуйте позже.",
        )


# ======================================================================
# Создание и запуск приложения
# ======================================================================

def _build_httpx_request(proxy_url: Optional[str]) -> HTTPXRequest:
    """Создаёт HTTPXRequest с trust_env=False.

    КРИТИЧЕСКИ ВАЖНО: trust_env=False
    Если в Windows/Linux/macOS стоят переменные http_proxy / https_proxy /
    no_proxy, то trust_env=True (по умолчанию) заставит httpx ИГНОРИРОВАТЬ
    наш прокси или использовать системный вместо явного.

    Это перекрёстная находка из VPg02 (pinned.trust_env = False):
    https://github.com/python-telegram-bot/python-telegram-bot/issues/4405
    """
    return HTTPXRequest(
        proxy=proxy_url,
        httpx_kwargs={"trust_env": False},
    )


async def _post_init_diagnostics(app: Application) -> None:
    """Диагностика подключения к Telegram API (async, вызывается post_init).

    Аналог bot.get_me() из VPg02: реальный запрос к api.telegram.org,
    который подтверждает, что токен и прокси работают.
    """
    try:
        bot_info = await app.bot.get_me()
        print(f"  ✓ Подключён как @{bot_info.username}")
    except Exception as exc:
        print(f"  ✗ Ошибка подключения к Telegram API: {exc}")
        print("  Проверьте TELEGRAM_BOT_TOKEN и PROXY_URL")


def create_bot_application(config: Config) -> Application:
    """Создаёт приложение Telegram-бота.

    Args:
        config: Конфигурация приложения.

    Returns:
        Приложение telegram-bot.
    """
    # Формируем прокси для Telegram
    telegram_proxy_url: Optional[str] = None
    if config.proxy_url:
        # httpx и python-telegram-bot требуют схему
        if not config.proxy_url.lower().startswith(("http://", "https://")):
            telegram_proxy_url = "http://" + config.proxy_url
        else:
            telegram_proxy_url = config.proxy_url

    # Создаём приложение
    builder = Application.builder()
    builder.token(config.telegram_bot_token)

    # КРИТИЧЕСКИ ВАЖНО: trust_env=False в обоих request-объектах
    # (обычный и get_updates). Без этого прокси может игнорироваться
    # из-за переменных окружения.
    builder.request(_build_httpx_request(telegram_proxy_url))
    builder.get_updates_request(_build_httpx_request(telegram_proxy_url))

    # skip_pending=True: не обрабатывать сообщения, полученные
    # пока бот был перезагружен (аналог skip_pending=True в VPg02).
    builder.post_init(_post_init_diagnostics)

    async def _post_shutdown(app: Application) -> None:
        logger.info("Приложение остановлено")

    builder.post_shutdown(_post_shutdown)

    application = builder.build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", _start))
    application.add_handler(CommandHandler("add_url", _add_url))
    application.add_handler(CommandHandler("search", _search))

    # Обработчик обычных сообщений (вопросов)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
    )

    # Обработчик ошибок
    application.add_error_handler(_error_handler)

    return application


def run_bot(config: Config) -> None:
    """Запускает Telegram-бота.

    Args:
        config: Конфигурация приложения.
    """
    if not config.telegram_bot_token:
        print("Ошибка: TELEGRAM_BOT_TOKEN не указан в .env")
        print("Получите токен у @BotFather в Telegram и добавьте в .env")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  RAG-Telegram-бот запускается...")
    print("=" * 60)

    if config.proxy_url:
        print("  Прокси для Telegram: включён")
    else:
        print("  Прокси для Telegram: не указан")

    print("  Бот готов к работе. Отправьте /start для проверки.\n")

    application = create_bot_application(config)

    # long_polling — более стабильный, чем default polling (аналог
    # bot.infinity_polling с long_polling_timeout из VPg02).
    # drop_pending_updates=True: не обрабатывать сообщения, полученные
    # пока бот был перезагружен (аналог skip_pending=True в VPg02).
    print("  Ожидаю сообщения...\n")
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            timeout=30,
        )
    except KeyboardInterrupt:
        print("\n  Остановка бота (Ctrl+C).")

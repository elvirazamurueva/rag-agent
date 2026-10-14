"""Конфигурация приложения RAG-ассистента.

Загружает переменные окружения из .env, проверяет обязательные настройки
и предоставляет безопасный доступ к параметрам.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Загружаем .env из текущей директории (если существует)
load_dotenv()


@dataclass(frozen=True)
class Config:
    """Неизменяемая конфигурация приложения."""

    # --- Обязательные поля ---
    genapi_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str

    # --- GenAPI: языковая модель (чат, OpenAI-совместимый прокси) ---
    genapi_url: str = "https://proxy.gen-api.ru/v1/chat/completions"
    model_id: str = "gpt-4o-mini"

    # --- GenAPI: эмбеддинги (нативный API генераций) ---
    genapi_embeddings_url: str = "https://api.gen-api.ru/api/v1/networks/embeddings"
    embedding_model: str = "text-embedding-3-small"

    # --- Pinecone ---
    pinecone_proxy_url: Optional[str] = None

    # --- Telegram-бот ---
    telegram_bot_token: Optional[str] = None
    proxy_url: Optional[str] = None

    # --- Таймауты (секунды) ---
    timeout_genapi: int = 30
    timeout_pinecone: int = 30

    def __post_init__(self) -> None:
        """Проверка обязательных параметров после создания."""
        required: list[tuple[str, str]] = [
            ("genapi_api_key", "GENAPI_API_KEY"),
            ("pinecone_api_key", "PINECONE_API_KEY"),
            ("pinecone_index_name", "PINECONE_INDEX_NAME"),
        ]
        missing = [
            (attr, env) for attr, env in required
            if not getattr(self, attr, None)
        ]
        if missing:
            names = ", ".join(env for _, env in missing)
            raise ValueError(
                f"Отсутствуют обязательные переменные окружения: {names}. "
                "Укажите их в файле .env."
            )


def load_config() -> Config:
    """Загружает и валидирует конфигурацию из переменных окружения.

    Returns:
        Config: настроенный объект конфигурации.

    Raises:
        ValueError: если отсутствуют обязательные переменные.
    """
    return Config(
        genapi_api_key=os.getenv("GENAPI_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", ""),
        genapi_url=os.getenv(
            "GENAPI_URL", "https://proxy.gen-api.ru/v1/chat/completions"
        ),
        model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
        genapi_embeddings_url=os.getenv(
            "GENAPI_EMBEDDINGS_URL",
            "https://api.gen-api.ru/api/v1/networks/embeddings",
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        pinecone_proxy_url=os.getenv("PINECONE_PROXY_URL", None),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", None),
        proxy_url=os.getenv("PROXY_URL", None),
        timeout_genapi=int(os.getenv("TIMEOUT_GENAPI", "30")),
        timeout_pinecone=int(os.getenv("TIMEOUT_PINECONE", "30")),
    )

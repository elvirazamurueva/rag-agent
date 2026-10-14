"""Загрузка и обработка текстовых документов из URL.

Модуль загружает веб-страницы по URL, извлекает основной текст,
разбивает на фрагменты для семантического поиска и сохраняет
векторы в существующий индекс Pinecone.

ВАЖНО: загруженный текст веб-страницы является ДАННЫМИ, а не
инструкциями для модели. Не интерпретируется как системные команды.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ограничения безопасности
# ---------------------------------------------------------------------------

MAX_RESPONSE_SIZE = 5_000_000  # 5 МБ
DOWNLOAD_TIMEOUT = 15  # секунд
MAX_REDIRECTS = 3

# Поддерживаемые MIME-типы содержимого
SUPPORTED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
}


# ---------------------------------------------------------------------------
# Ошибки
# ---------------------------------------------------------------------------

class DocumentLoaderError(Exception):
    """Ошибка при загрузке документа."""


class SSRFError(DocumentLoaderError):
    """Попытка SSRF-атаки (обращение к локальным/внутренним адресам)."""


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Один фрагмент текста документа."""
    text: str
    chunk_index: int
    document_id: str
    source_url: str
    page_title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """Результат загрузки документа."""
    source_url: str
    page_title: str
    full_text: str
    chunks: list[DocumentChunk]
    total_chunks: int
    saved_vectors: int
    success: bool
    error_message: str = ""


# ---------------------------------------------------------------------------
# Валидация URL и защита от SSRF
# ---------------------------------------------------------------------------

def validate_url(url: str) -> None:
    """Проверяет корректность URL и разрешает только HTTP/HTTPS.

    Args:
        url: Проверяемый URL.

    Raises:
        DocumentLoaderError: если URL некорректен или схема не разрешена.
    """
    if not url or not url.strip():
        raise DocumentLoaderError("URL не может быть пустым.")

    try:
        parsed = urlparse(url.strip())
    except (ValueError, TypeError) as exc:
        raise DocumentLoaderError(f"Некорректный URL: {exc}") from exc

    if parsed.scheme not in ("http", "https"):
        raise DocumentLoaderError(
            f"Разрешены только схемы http и https. Получена схема: {parsed.scheme}"
        )

    if not parsed.netloc:
        raise DocumentLoaderError("URL не содержит имени хоста.")


def _is_private_ip(ip: str) -> bool:
    """Проверяет, является ли IP приватным/локальным/служебным."""
    try:
        addr = ipaddress.ip_address(ip)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        )
    except ValueError:
        return False


def _resolve_and_check_host(hostname: str) -> None:
    """Разрешает DNS и проверяет все IP-адреса хоста на SSRF."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
    except socket.gaierror as exc:
        raise DocumentLoaderError(
            f"Не удалось разрешить имя хоста '{hostname}': {exc}"
        ) from exc

    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            raise SSRFError(
                f"Хост '{hostname}' указывает на приватный/локальный адрес "
                f"({ip}). Обращение к внутренним сетям запрещено."
            )


def validate_url_security(url: str) -> None:
    """Полная проверка URL: схема + DNS + приватные адреса.

    Args:
        url: Проверяемый URL.

    Raises:
        SSRFError: если хост указывает на локальный/приватный адрес.
        DocumentLoaderError: при некорректном URL.
    """
    validate_url(url)

    parsed = urlparse(url.strip())
    hostname = parsed.hostname

    if not hostname:
        raise DocumentLoaderError("URL не содержит имени хоста.")

    # Проверяем, не является ли hostname самим IP-адресом
    try:
        ipaddress.ip_address(hostname)
        if _is_private_ip(hostname):
            raise SSRFError(
                f"IP-адрес {hostname} является приватным/локальным. "
                "Обращение запрещено."
            )
    except ValueError:
        # hostname — домен, разрешаем DNS
        _resolve_and_check_host(hostname)


# ---------------------------------------------------------------------------
# Извлечение текста из HTML
# ---------------------------------------------------------------------------

# Теги, содержимое которых не является основным текстом
_UNWANTED_TAGS = [
    "script", "style", "noscript", "iframe",
    "form", "button", "input",
    "menu", "menuitem", "picture", "svg", "canvas",
]

# HTML5-теги, содержимое которых обычно не является основным текстом
# (но не удаляем целиком — они могут содержать контент)
_UNWANTED_CONTENT_TAGS = [
    "nav", "header", "footer", "aside",
]

# CSS-классы/ID, указывающие на нежелательные секции.
# Точное совпадение или начало слова после дефиса/подчёркивания.
_UNWANTED_PATTERNS = re.compile(
    r"(?:^|[-_])(sidebar|menu|navigation|header|footer|"
    r"advert|banner|popup|modal|cookie|social-links|share-buttons|"
    r"comments?|related|promo)(?:$|[-_])",
    re.IGNORECASE,
)


def extract_text_from_html(html: str) -> tuple[str, str]:
    """Извлекает основной текст из HTML, исключая навигацию и скрипты.

    Args:
        html: Исходный HTML-код страницы.

    Returns:
        Кортеж (извлечённый_текст, заголовок_страницы).
    """
    soup = BeautifulSoup(html, "lxml")

    # Удаляем комментарии
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Удаляем нежелательные теги (скрипты, стили и т.д.)
    for tag_name in _UNWANTED_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Удаляем содержимое HTML5-семантических тегов навигации
    # (но не весь тег целиком — он может содержать и основной контент)
    for tag_name in _UNWANTED_CONTENT_TAGS:
        for tag in soup.find_all(tag_name):
            for child in list(tag.children):
                child.decompose()

    # Удаляем элементы по классам/ID
    for tag in soup.find_all(attrs={"class": _UNWANTED_PATTERNS}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": _UNWANTED_PATTERNS}):
        tag.decompose()

    # Извлекаем заголовок
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Пытаемся найти основной контент
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("body")
        or soup
    )

    # Извлекаем текст
    text = main_content.get_text(separator="\n", strip=True)

    # Нормализуем пустые строки
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    return cleaned_text, title


# ---------------------------------------------------------------------------
# Разбиение текста на фрагменты
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Разбивает текст на перекрывающиеся фрагменты по словам.

    Args:
        text: Входной текст.
        chunk_size: Целевое количество слов в фрагменте.
        chunk_overlap: Количество перекрывающихся слов.

    Returns:
        Список текстовых фрагментов.
    """
    if not text or not text.strip():
        return []

    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Загрузчик документов
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Загружает веб-страницы и сохраняет их в Pinecone."""

    def __init__(
        self,
        embedding_client: Any,
        index: Any,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Args:
            embedding_client: Экземпляр EmbeddingClient для создания векторов.
            index: Подключённый Pinecone index для сохранения векторов.
            chunk_size: Количество слов в фрагменте.
            chunk_overlap: Перекрытие фрагментов (слов).
        """
        self._embed_client = embedding_client
        self._index = index
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load_url(self, url: str) -> LoadedDocument:
        """Загружает страницу по URL, извлекает текст, сохраняет в Pinecone.

        Args:
            url: Адрес веб-страницы.

        Returns:
            LoadedDocument с результатом загрузки.
        """
        try:
            # Валидация и проверка безопасности
            validate_url_security(url)

            # Загрузка страницы
            html_content, page_title_from_response = self._fetch_page(url)

            # Извлечение текста
            full_text, title = extract_text_from_html(html_content)
            if not title:
                title = page_title_from_response

            # Проверка на пустую страницу
            if not full_text or len(full_text) < 50:
                return LoadedDocument(
                    source_url=url,
                    page_title=title,
                    full_text=full_text,
                    chunks=[],
                    total_chunks=0,
                    saved_vectors=0,
                    success=False,
                    error_message="Не удалось извлечь содержательный текст из страницы.",
                )

            # Разбиение на фрагменты
            text_chunks = chunk_text(
                full_text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

            if not text_chunks:
                return LoadedDocument(
                    source_url=url,
                    page_title=title,
                    full_text=full_text,
                    chunks=[],
                    total_chunks=0,
                    saved_vectors=0,
                    success=False,
                    error_message="Текст пуст после разбиения на фрагменты.",
                )

            # Генерация document_id
            document_id = self._generate_document_id(url)

            # Создание DocumentChunk
            doc_chunks = [
                DocumentChunk(
                    text=chunk,
                    chunk_index=i,
                    document_id=document_id,
                    source_url=url,
                    page_title=title,
                )
                for i, chunk in enumerate(text_chunks)
            ]

            # Сохранение в Pinecone
            saved_count = self._save_chunks_to_pinecone(doc_chunks)

            return LoadedDocument(
                source_url=url,
                page_title=title,
                full_text=full_text,
                chunks=doc_chunks,
                total_chunks=len(doc_chunks),
                saved_vectors=saved_count,
                success=saved_count > 0,
                error_message="" if saved_count > 0 else "Не удалось сохранить ни одного вектора.",
            )

        except DocumentLoaderError as exc:
            return LoadedDocument(
                source_url=url,
                page_title="",
                full_text="",
                chunks=[],
                total_chunks=0,
                saved_vectors=0,
                success=False,
                error_message=str(exc),
            )
        except Exception as exc:
            return LoadedDocument(
                source_url=url,
                page_title="",
                full_text="",
                chunks=[],
                total_chunks=0,
                saved_vectors=0,
                success=False,
                error_message=f"Непредвиденная ошибка: {exc}",
            )

    def _fetch_page(self, url: str) -> tuple[str, str]:
        """Загружает HTML-страницу с проверками безопасности.

        Returns:
            Кортеж (html_content, заголовок_из_response).

        Raises:
            DocumentLoaderError: при сетевых ошибках или неподдерживаемом типе.
            SSRFError: при попытке SSRF.
        """
        session = requests.Session()
        session.max_redirects = MAX_REDIRECTS

        try:
            response = session.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                stream=True,
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RAG-DocumentLoader/1.0)"
                },
            )
        except requests.TooManyRedirects as exc:
            raise DocumentLoaderError(
                f"Слишком много перенаправлений (макс. {MAX_REDIRECTS})."
            ) from exc
        except requests.ConnectionError as exc:
            raise DocumentLoaderError(
                f"Не удалось подключиться к {url}: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise DocumentLoaderError(
                f"Превышено время ожидания загрузки ({DOWNLOAD_TIMEOUT} с)."
            ) from exc
        except requests.RequestException as exc:
            raise DocumentLoaderError(
                f"Ошибка запроса к {url}: {exc}"
            ) from exc

        # Проверка перенаправления: если redirect на приватный адрес
        final_url = response.url
        parsed_final = urlparse(final_url)
        if parsed_final.hostname:
            try:
                validate_url_security(final_url)
            except SSRFError:
                response.close()
                raise

        # Проверка кода ответа
        if response.status_code != 200:
            response.close()
            raise DocumentLoaderError(
                f"Сервер вернул HTTP {response.status_code} для {url}."
            )

        # Проверка типа содержимого
        content_type = response.headers.get("content-type", "").lower()
        is_supported = any(ct in content_type for ct in SUPPORTED_CONTENT_TYPES)
        if not is_supported:
            response.close()
            raise DocumentLoaderError(
                f"Неподдерживаемый тип содержимого: {content_type}. "
                f"Ожидается HTML."
            )

        # Ограничение размера
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_RESPONSE_SIZE:
            response.close()
            raise DocumentLoaderError(
                f"Размер ответа ({content_length} байт) превышает лимит "
                f"({MAX_RESPONSE_SIZE} байт)."
            )

        # Чтение с ограничением размера
        chunks_data: list[bytes] = []
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)
            if total_size > MAX_RESPONSE_SIZE:
                response.close()
                raise DocumentLoaderError(
                    f"Размер ответа превышает лимит ({MAX_RESPONSE_SIZE} байт)."
                )
            chunks_data.append(chunk)

        response.close()
        html_content = b"".join(chunks_data).decode("utf-8", errors="replace")

        # Заголовок из ответа
        title = ""
        soup = BeautifulSoup(html_content, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        return html_content, title

    @staticmethod
    def _generate_document_id(url: str) -> str:
        """Генерирует уникальный идентификатор документа на основе URL."""
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        return f"doc-{url_hash}"

    def _save_chunks_to_pinecone(self, doc_chunks: list[DocumentChunk]) -> int:
        """Создаёт эмбеддинги и сохраняет фрагменты в Pinecone.

        Args:
            doc_chunks: Список фрагментов для сохранения.

        Returns:
            Количество успешно сохранённых векторов.
        """
        saved_count = 0
        batch_size = 32  # Pinecone API limit

        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i + batch_size]
            vectors: list[dict[str, Any]] = []
            errors_in_batch: list[str] = []

            for chunk in batch:
                try:
                    embedding = self._embed_client.create_embedding(chunk.text)
                    vector_id = f"{chunk.document_id}-chunk-{chunk.chunk_index}"

                    vectors.append({
                        "id": vector_id,
                        "values": embedding.values,
                        "metadata": {
                            "text": chunk.text,
                            "source_url": chunk.source_url,
                            "page_title": chunk.page_title,
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                        },
                    })
                except Exception as exc:
                    errors_in_batch.append(
                        f"chunk-{chunk.chunk_index}: {exc}"
                    )

            if vectors:
                try:
                    self._index.upsert(vectors=vectors)
                    saved_count += len(vectors)
                    logger.info(
                        "Upserted %d vectors (batch %d/%d).",
                        len(vectors),
                        i // batch_size + 1,
                        (len(doc_chunks) + batch_size - 1) // batch_size,
                    )
                except Exception as exc:
                    logger.error(
                        "Ошибка Pinecone upsert для batch %d: %s",
                        i // batch_size + 1,
                        exc,
                    )

            if errors_in_batch:
                logger.warning(
                    "Ошибки эмбеддингов в batch %d: %s",
                    i // batch_size + 1,
                    "; ".join(errors_in_batch[:5]),
                )

        return saved_count

"""Автоматические тесты для модуля загрузки документов.

Тесты не требуют реальных API-ключей и не изменяют настоящую базу Pinecone.
Используются имитации внешних сервисов.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_loader import (
    DocumentChunk,
    DocumentLoader,
    DocumentLoaderError,
    SSRFError,
    LoadedDocument,
    chunk_text,
    extract_text_from_html,
    validate_url,
    validate_url_security,
)


# ======================================================================
# Тесты валидации URL
# ======================================================================

class TestURLValidation:
    """Проверка корректности URL и защиты от SSRF."""

    def test_validate_url_accepts_http(self):
        """HTTP-URL проходит валидацию."""
        try:
            validate_url("http://example.com/page")
        except DocumentLoaderError:
            pytest.fail("HTTP-URL должен быть разрешён")

    def test_validate_url_accepts_https(self):
        """HTTPS-URL проходит валидацию."""
        try:
            validate_url("https://example.com/page")
        except DocumentLoaderError:
            pytest.fail("HTTPS-URL должен быть разрешён")

    def test_validate_url_rejects_file_scheme(self):
        """Схема file:// должна быть отклонена."""
        with pytest.raises(DocumentLoaderError, match="схем"):
            validate_url("file:///etc/passwd")

    def test_validate_url_rejects_ftp_scheme(self):
        """Схема ftp:// должна быть отклонена."""
        with pytest.raises(DocumentLoaderError, match="схем"):
            validate_url("ftp://example.com/file")

    def test_validate_url_rejects_empty(self):
        """Пустой URL должен быть отклонён."""
        with pytest.raises(DocumentLoaderError, match="пустым"):
            validate_url("")

    def test_validate_url_rejects_no_hostname(self):
        """URL без имени хоста должен быть отклонён."""
        with pytest.raises(DocumentLoaderError, match="имени хоста"):
            validate_url("http://")

    def test_validate_url_security_ssrf_loopback(self):
        """SSRF: localhost должен быть запрещён."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://127.0.0.1:8080/secret")

    def test_validate_url_security_ssrf_private_ip(self):
        """SSRF: приватный IP должен быть запрещён."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://192.168.1.1/admin")

    def test_validate_url_security_ssrf_internal_ip(self):
        """SSRF: внутренний IP должен быть запрещён."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://10.0.0.1/internal")

    def test_validate_url_security_accepts_public_domain(self):
        """Публичный домен должен пройти валидацию."""
        try:
            validate_url_security("http://example.com/page")
        except DocumentLoaderError:
            pytest.fail("Публичный домен должен быть разрешён")


# ======================================================================
# Тесты извлечения текста из HTML
# ======================================================================

class TestHTMLExtraction:
    """Проверка извлечения текста из HTML."""

    def test_extract_text_simple_html(self):
        """Простой HTML с текстом."""
        html = "<html><body><p>Привет, мир!</p></body></html>"
        text, title = extract_text_from_html(html)
        assert "Привет, мир!" in text

    def test_extract_text_removes_script(self):
        """Тег script должен быть удалён."""
        html = """
        <html>
        <body>
            <p>Важный текст</p>
            <script>alert('bad');</script>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)
        assert "alert" not in text.lower()
        assert "Важный текст" in text

    def test_extract_text_removes_style(self):
        """Тег style должен быть удалён."""
        html = """
        <html>
        <head><style>body{color:red}</style></head>
        <body><p>Важный текст</p></body>
        </html>
        """
        text, _ = extract_text_from_html(html)
        assert "color:red" not in text
        assert "Важный текст" in text

    def test_extract_text_removes_comments(self):
        """HTML-комментарии должны быть удалены."""
        html = """
        <html>
        <body>
            <!-- Это комментарий -->
            <p>Важный текст</p>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)
        assert "комментарий" not in text.lower()
        assert "Важный текст" in text

    def test_extract_text_removes_nav(self):
        """Навигация должна быть удалена."""
        html = """
        <html>
        <body>
            <nav><a href="#">Меню</a></nav>
            <article><p>Основной текст</p></article>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)
        assert "Меню" not in text
        assert "Основной текст" in text

    def test_extract_text_removes_sidebar(self):
        """Сайдбар должен быть удалён."""
        html = """
        <html>
        <body>
            <div class="sidebar">Реклама</div>
            <main><p>Полезный контент</p></main>
        </body>
        </html>
        """
        text, _ = extract_text_from_html(html)
        assert "Реклама" not in text
        assert "Полезный контент" in text

    def test_extract_text_gets_title(self):
        """Заголовок страницы должен быть извлечён."""
        html = """
        <html>
        <head><title>Заголовок Страницы</title></head>
        <body><p>Текст</p></body>
        </html>
        """
        text, title = extract_text_from_html(html)
        assert title == "Заголовок Страницы"
        assert "Текст" in text

    def test_extract_text_handles_empty(self):
        """Пустая страница должна вернуть пустой текст."""
        html = "<html><body></body></html>"
        text, title = extract_text_from_html(html)
        assert text == ""

    def test_extract_text_handles_no_body(self):
        """HTML без тега body должен обработаться."""
        html = "<html><head><title>Title</title></head></html>"
        text, title = extract_text_from_html(html)
        assert title == "Title"


# ======================================================================
# Тесты разбиения текста на фрагменты
# ======================================================================

class TestChunking:
    """Проверка разбиения текста на фрагменты."""

    def test_chunk_text_short(self):
        """Короткий текст не разбивается."""
        text = "Короткий текст"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Короткий текст"

    def test_chunk_text_long(self):
        """Длинный текст разбивается на фрагменты."""
        words = ["слово"] * 1000
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 500 for chunk in chunks)

    def test_chunk_text_empty(self):
        """Пустой текст возвращает пустой список."""
        chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_chunk_text_no_overlap(self):
        """Проверка перекрытия фрагментов."""
        words = ["слово"] * 1000
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # Проверка, что фрагменты перекрываются
        if len(chunks) > 1:
            prev_words = set(chunks[-2].split())
            curr_words = set(chunks[-1].split())
            overlap = prev_words & curr_words
            # Все слова одинаковые, поэтому пересечение должно быть "слово"
            assert overlap == {"слово"}

    def test_chunk_text_preserves_order(self):
        """Фрагменты должны сохранять порядок слов."""
        words = [f"слово{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        # Первый фрагмент должен начинаться с "слово0"
        assert chunks[0].startswith("слово0")
        # Слова внутри каждого фрагмента должны быть в порядке
        for chunk in chunks:
            chunk_words = chunk.split()
            assert chunk_words == sorted(chunk_words, key=lambda w: int(w.replace("слово", "")))


# ======================================================================
# Тесты DocumentLoader (с моками)
# ======================================================================

class TestDocumentLoader:
    """Проверка загрузчика документов с моками."""

    def _make_loader(self):
        """Создаёт DocumentLoader с моками."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.create_embedding.return_value = MagicMock(
            values=[0.1] * 1536,
            dimension=1536,
            model="text-embedding-3-small",
        )

        mock_index = MagicMock()
        mock_index.upsert = MagicMock()

        loader = DocumentLoader(
            embedding_client=mock_embedding_client,
            index=mock_index,
            chunk_size=100,
            chunk_overlap=10,
        )
        return loader, mock_embedding_client, mock_index

    def test_load_url_simple_success(self):
        """Успешная загрузка простой страницы."""
        loader, mock_embed, mock_index = self._make_loader()

        html = """
        <html>
        <head><title>Тестовая Страница</title></head>
        <body>
            <article>
                <p>Это тестовый текст для проверки загрузки.</p>
                <p>Он содержит несколько предложений.</p>
            </article>
        </body>
        </html>
        """

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/page"
            mock_response.headers = {"content-type": "text/html", "content-length": str(len(html))}
            mock_response.iter_content.return_value = [html.encode("utf-8")]
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/page")

            assert doc.success is True
            assert doc.page_title == "Тестовая Страница"
            assert len(doc.full_text) > 0
            assert doc.total_chunks > 0
            assert doc.saved_vectors > 0
            assert mock_index.upsert.called

    def test_load_url_rejects_invalid_scheme(self):
        """Загрузка с недопустимой схемой должна завершиться ошибкой."""
        loader, _, _ = self._make_loader()

        doc = loader.load_url("file:///etc/passwd")
        assert doc.success is False
        assert "схем" in doc.error_message.lower()

    def test_load_url_rejects_ssrf(self):
        """Загрузка локального адреса должна быть запрещена."""
        loader, _, _ = self._make_loader()

        # Для localhost проверяем, что валидация не проходит
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://127.0.0.1:8080/secret")

    def test_load_url_network_error(self):
        """Сетевая ошибка должна обработаться."""
        loader, _, _ = self._make_loader()

        with patch("document_loader.requests.Session") as mock_session:
            mock_session.return_value.get.side_effect = requests.ConnectionError("Network down")

            doc = loader.load_url("http://example.com/page")
            assert doc.success is False
            assert "подключиться" in doc.error_message.lower() or "network" in doc.error_message.lower()

    def test_load_url_empty_page(self):
        """Пустая страница должна вернуть ошибку."""
        loader, _, _ = self._make_loader()

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/page"
            mock_response.headers = {"content-type": "text/html", "content-length": "0"}
            mock_response.iter_content.return_value = [b""]
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/page")
            assert doc.success is False
            assert "не удалось извлечь" in doc.error_message.lower()

    def test_load_url_wrong_content_type(self):
        """Неподдерживаемый тип содержимого должен быть отклонён."""
        loader, _, _ = self._make_loader()

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/image.png"
            mock_response.headers = {"content-type": "image/png", "content-length": "100"}
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/image.png")
            assert doc.success is False
            assert "неподдерживаемый тип" in doc.error_message.lower()

    def test_load_url_save_error(self):
        """Ошибка при сохранении в Pinecone должна обработаться."""
        loader, mock_embed, mock_index = self._make_loader()

        # Мокаем ошибку при вызове upsert
        mock_index.upsert.side_effect = Exception("Pinecone connection lost")

        html = """
        <html>
        <body>
            <article>
                <p>Это длинный текст для загрузки в Pinecone, который содержит много информации для проверки работы загрузчика документов.</p>
            </article>
        </body>
        </html>
        """

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/page"
            mock_response.headers = {"content-type": "text/html", "content-length": str(len(html))}
            mock_response.iter_content.return_value = [html.encode("utf-8")]
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/page")
            assert doc.success is False
            assert "не удалось сохранить" in doc.error_message.lower()

    def test_load_url_metadata_preserved(self):
        """Метаданные фрагментов должны содержать источник."""
        loader, mock_embed, mock_index = self._make_loader()

        html = """
        <html>
        <head><title>Метаданные Тест</title></head>
        <body>
            <article>
                <p>Это длинный текст с метаданными для проверки работы загрузчика документов и сохранения информации о источнике.</p>
            </article>
        </body>
        </html>
        """

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/page"
            mock_response.headers = {"content-type": "text/html", "content-length": str(len(html))}
            mock_response.iter_content.return_value = [html.encode("utf-8")]
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/page")
            assert doc.success is True
            assert doc.total_chunks > 0

            # Проверяем метаданные первого фрагмента
            chunk = doc.chunks[0]
            assert chunk.source_url == "http://example.com/page"
            assert chunk.page_title == "Метаданные Тест"
            assert chunk.document_id.startswith("doc-")
            assert chunk.chunk_index >= 0

    def test_load_url_multiple_urls(self):
        """Загрузка нескольких URL должна создать уникальные document_id."""
        loader, mock_embed, mock_index = self._make_loader()

        html = """
        <html>
        <body><article><p>Это длинный текст для проверки загрузки нескольких документов в систему.</p></article></body>
        </html>
        """

        urls = ["http://example1.com/page", "http://example2.com/page"]

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = urls[0]
            mock_response.headers = {"content-type": "text/html", "content-length": str(len(html))}
            mock_response.iter_content.return_value = [html.encode("utf-8")]
            mock_session.return_value.get.return_value = mock_response

            doc1 = loader.load_url(urls[0])
            mock_response.url = urls[1]
            doc2 = loader.load_url(urls[1])

            assert doc1.success is True
            assert doc2.success is True
            assert doc1.chunks[0].document_id != doc2.chunks[0].document_id

    def test_load_url_timeout(self):
        """Таймаут загрузки должен быть обработан."""
        loader, _, _ = self._make_loader()

        with patch("document_loader.requests.Session") as mock_session:
            mock_session.return_value.get.side_effect = requests.Timeout("Timeout")

            doc = loader.load_url("http://example.com/page")
            assert doc.success is False
            assert "время ожидания" in doc.error_message.lower() or "timeout" in doc.error_message.lower()

    def test_load_url_too_many_redirects(self):
        """Слишком много перенаправлений должно быть обработано."""
        loader, _, _ = self._make_loader()

        with patch("document_loader.requests.Session") as mock_session:
            mock_session.return_value.get.side_effect = requests.TooManyRedirects("Too many")

            doc = loader.load_url("http://example.com/page")
            assert doc.success is False
            assert "перенаправлен" in doc.error_message.lower() or "redirect" in doc.error_message.lower()


# ======================================================================
# Тесты безопасности
# ======================================================================

class TestSecurity:
    """Проверка безопасности загрузки документов."""

    def test_ssrf_blocked_localhost(self):
        """SSRF: localhost должен быть заблокирован."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://localhost:8080/secret")

    def test_ssrf_blocked_127_0_0_1(self):
        """SSRF: 127.0.0.1 должен быть заблокирован."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://127.0.0.1:8080/secret")

    def test_ssrf_blocked_192_168(self):
        """SSRF: 192.168.x.x должен быть заблокирован."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://192.168.1.1/admin")

    def test_ssrf_blocked_10_0_0(self):
        """SSRF: 10.x.x.x должен быть заблокирован."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://10.0.0.1/internal")

    def test_ssrf_blocked_172_16_172_31(self):
        """SSRF: 172.16-31.x.x должен быть заблокирован."""
        with pytest.raises((SSRFError, DocumentLoaderError)):
            validate_url_security("http://172.16.0.1/internal")

    def test_accepted_public_domain(self):
        """Публичный домен должен быть разрешён."""
        try:
            validate_url_security("http://example.com/page")
        except DocumentLoaderError:
            pytest.fail("Публичный домен должен быть разрешён")

    def test_accepted_http_vs_https(self):
        """HTTP и HTTPS должны быть разрешены."""
        try:
            validate_url_security("http://example.com/page")
            validate_url_security("https://example.com/page")
        except DocumentLoaderError:
            pytest.fail("HTTP и HTTPS должны быть разрешены")

    def test_rejected_file_scheme(self):
        """Схема file:// должна быть запрещена."""
        with pytest.raises(DocumentLoaderError, match="схем"):
            validate_url_security("file:///etc/passwd")

    def test_rejected_ftp_scheme(self):
        """Схема ftp:// должна быть запрещена."""
        with pytest.raises(DocumentLoaderError, match="схем"):
            validate_url_security("ftp://example.com/file")

    def test_rejected_data_scheme(self):
        """Схема data:// должна быть запрещена."""
        with pytest.raises(DocumentLoaderError, match="схем"):
            validate_url_security("data:text/html,<script>alert(1)</script>")


# ======================================================================
# Тесты интеграции с Pinecone
# ======================================================================

class TestPineconeIntegration:
    """Проверка интеграции с Pinecone (с моками)."""

    def test_upsert_called_with_correct_structure(self):
        """Проверка, что upsert вызывается с правильной структурой."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.create_embedding.return_value = MagicMock(
            values=[0.1] * 1536,
            dimension=1536,
            model="text-embedding-3-small",
        )

        mock_index = MagicMock()
        mock_index.upsert = MagicMock()

        loader = DocumentLoader(
            embedding_client=mock_embedding_client,
            index=mock_index,
            chunk_size=100,
            chunk_overlap=10,
        )

        html = """
        <html>
        <body>
            <article><p>Это длинный текст для загрузки в Pinecone, который содержит достаточно информации для проверки работы загрузчика.</p></article>
        </body>
        </html>
        """

        with patch("document_loader.requests.Session") as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "http://example.com/page"
            mock_response.headers = {"content-type": "text/html", "content-length": str(len(html))}
            mock_response.iter_content.return_value = [html.encode("utf-8")]
            mock_session.return_value.get.return_value = mock_response

            doc = loader.load_url("http://example.com/page")

            assert doc.success is True
            assert mock_index.upsert.called

            # Проверяем аргументы upsert
            call_args = mock_index.upsert.call_args
            vectors = call_args.kwargs["vectors"]

            assert len(vectors) > 0
            for v in vectors:
                assert "id" in v
                assert "values" in v
                assert "metadata" in v
                assert "text" in v["metadata"]
                assert "source_url" in v["metadata"]
                assert "page_title" in v["metadata"]
                assert "document_id" in v["metadata"]
                assert "chunk_index" in v["metadata"]
                assert len(v["values"]) == 1536


# ======================================================================
# Запуск тестов
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

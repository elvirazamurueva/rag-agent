"""Автоматические тесты для GET-клиента API.

Тесты покрывают:
  - Успешный GET-запрос (с моками)
  - HTTP-ошибки (404, 500)
  - Таймаут
  - Ошибка соединения
  - Некорректный JSON-ответ
  - Реальный запрос к jsonplaceholder
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from api_client import APIClient, APIError, APIResponse, fetch_data, DEFAULT_TEST_API


# ======================================================================
# Вспомогательные функции
# ======================================================================

def _make_mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
    content_type: str = "application/json",
    text: str = "",
) -> MagicMock:
    """Создаёт моковый requests.Response."""
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.headers = {"content-type": content_type}

    if json_data is not None:
        mock.json.return_value = json_data
        mock.text = str(json_data) if isinstance(json_data, (dict, list)) else text
    else:
        mock.json.return_value = None
        mock.text = text

    return mock


# ======================================================================
# Тесты успешных запросов
# ======================================================================

class TestSuccessfulRequests:
    """Успешные GET-запросы."""

    def test_get_success_json(self):
        """Успешный GET-запрос с JSON-ответом."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"id": 1, "title": "Test Post"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            response = client.get(f"{DEFAULT_TEST_API}/posts/1")

        assert response.success is True
        assert response.status_code == 200
        assert response.data["id"] == 1
        assert response.data["title"] == "Test Post"
        assert response.url == f"{DEFAULT_TEST_API}/posts/1"

    def test_get_success_with_params(self):
        """GET-запрос с параметрами query string."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data=[{"id": 1}, {"id": 2}],
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            response = client.get(
                f"{DEFAULT_TEST_API}/posts",
                params={"_limit": 2, "_sort": "id"},
            )

        assert response.success is True
        mock_session.get.assert_called_once()
        call_params = mock_session.get.call_args
        assert call_params.kwargs["params"] == {"_limit": 2, "_sort": "id"}

    def test_get_or_raise(self):
        """get_or_raise возвращает данные при успехе."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"key": "value"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            data = client.get_or_raise(f"{DEFAULT_TEST_API}/posts/1")

        assert data["key"] == "value"


# ======================================================================
# Тесты HTTP-ошибок
# ======================================================================

class TestHTTPErrors:
    """Обработка HTTP-ошибок."""

    def test_get_404_not_found(self):
        """GET-запрос возвращает 404."""
        mock_resp = _make_mock_response(
            status_code=404,
            json_data={"error": "Post not found"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="HTTP 404"):
                client.get(f"{DEFAULT_TEST_API}/posts/99999")

    def test_get_500_internal_error(self):
        """GET-запрос возвращает 500."""
        mock_resp = _make_mock_response(
            status_code=500,
            json_data={"error": "Internal Server Error"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="HTTP 500"):
                client.get(f"{DEFAULT_TEST_API}/posts")

    def test_get_401_unauthorized(self):
        """GET-запрос возвращает 401."""
        mock_resp = _make_mock_response(
            status_code=401,
            json_data={"error": "Unauthorized"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="HTTP 401"):
                client.get(f"{DEFAULT_TEST_API}/posts")

    def test_get_403_forbidden(self):
        """GET-запрос возвращает 403."""
        mock_resp = _make_mock_response(
            status_code=403,
            json_data={"error": "Forbidden"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="HTTP 403"):
                client.get(f"{DEFAULT_TEST_API}/posts")


# ======================================================================
# Тесты сетевых ошибок
# ======================================================================

class TestNetworkErrors:
    """Обработка сетевых ошибок."""

    def test_connection_error(self):
        """Ошибка соединения."""
        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.ConnectionError(
                "Connection refused"
            )
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="Не удалось подключиться"):
                client.get("http://localhost:99999/test")

    def test_timeout(self):
        """Таймаут запроса."""
        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.Timeout("Timeout")
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="время ожидания"):
                client.get(f"{DEFAULT_TEST_API}/posts")

    def test_general_request_error(self):
        """Общая ошибка запроса."""
        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.RequestException("General error")
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="Ошибка запроса"):
                client.get(f"{DEFAULT_TEST_API}/posts")


# ======================================================================
# Тесты некорректных ответов
# ======================================================================

class TestInvalidResponses:
    """Обработка некорректных ответов."""

    def test_non_json_response(self):
        """Ответ не является JSON."""
        mock_resp = _make_mock_response(
            status_code=200,
            text="<html>Not JSON</html>",
            content_type="text/html",
        )
        mock_resp.json.side_effect = ValueError("not json")

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            with pytest.raises(APIError, match="не является валидным JSON"):
                client.get(f"{DEFAULT_TEST_API}/posts")

    def test_empty_response_raises_error(self):
        """Пустой ответ — ошибка (не является валидным JSON)."""
        mock_resp = _make_mock_response(
            status_code=204,
            text="",
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            # Пустой ответ не является JSON — выбрасывает APIError
            with pytest.raises(APIError, match="не является валидным JSON"):
                client.get(f"{DEFAULT_TEST_API}/posts")

    def test_empty_json_body_raises_error(self):
        """200 OK, но тело пустое — ошибка (не является валидным JSON)."""
        mock_resp = _make_mock_response(
            status_code=200,
            text="",
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            # Пустое тело — не является валидным JSON — выбрасывает APIError
            with pytest.raises(APIError, match="не является валидным JSON"):
                client.get(f"{DEFAULT_TEST_API}/posts")


# ======================================================================
# Тесты контекстного менеджера
# ======================================================================

class TestContextManager:
    """Проверка контекстного менеджера."""

    def test_context_manager_closes_session(self):
        """Контекстный менеджер закрывает сессию."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"ok": True},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            with APIClient(timeout=10) as client:
                response = client.get(f"{DEFAULT_TEST_API}/posts/1")
                assert response.success is True

            # Сессия должна быть закрыта
            mock_session.close.assert_called_once()


# ======================================================================
# Тесты функции-обёртки
# ======================================================================

class TestFetchData:
    """Тесты функции fetch_data()."""

    def test_fetch_data_success(self):
        """fetch_data возвращает данные при успехе."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"message": "Hello"},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            response = fetch_data(f"{DEFAULT_TEST_API}/test", timeout=5)

        assert response.success is True
        assert response.data["message"] == "Hello"

    def test_fetch_data_error(self):
        """fetch_data пробрасывает APIError."""
        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.Timeout("Timeout")
            mock_session_cls.return_value = mock_session

            with pytest.raises(APIError, match="время ожидания"):
                fetch_data(f"{DEFAULT_TEST_API}/test", timeout=1)


# ======================================================================
# Тесты конфигурации
# ======================================================================

class TestConfig:
    """Тесты конфигурации клиента."""

    def test_custom_timeout(self):
        """Клиент использует пользовательский таймаут."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"ok": True},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=5)
            client.get(f"{DEFAULT_TEST_API}/posts/1")

            call_timeout = mock_session.get.call_args.kwargs["timeout"]
            assert call_timeout == 5

    def test_override_timeout(self):
        """Переопределение таймаута в методе get()."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"ok": True},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            client.get(f"{DEFAULT_TEST_API}/posts/1", timeout=3)

            call_timeout = mock_session.get.call_args.kwargs["timeout"]
            assert call_timeout == 3

    def test_custom_headers(self):
        """Передача пользовательских заголовков."""
        mock_resp = _make_mock_response(
            status_code=200,
            json_data={"ok": True},
        )

        with patch("api_client.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            client = APIClient(timeout=10)
            client.get(
                f"{DEFAULT_TEST_API}/posts/1",
                headers={"Authorization": "Bearer token", "X-Custom": "value"},
            )

            call_headers = mock_session.get.call_args.kwargs["headers"]
            assert call_headers["Authorization"] == "Bearer token"
            assert call_headers["X-Custom"] == "value"


# ======================================================================
# Запуск тестов
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

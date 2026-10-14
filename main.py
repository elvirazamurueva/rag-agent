"""Диагностический запуск RAG-ассистента.

Проверяет по шагам:
  1. Загрузка конфигурации
  2. Подключение к Pinecone
  3. Проверка индекса (существование, размерность, количество векторов)
  4. Создание эмбеддинга тестового вопроса
  5. Поиск в Pinecone
  6. Вывод диагностического отчёта

Также поддерживает команды:
  python main.py load <url1> [url2] ...  — загрузка документов из URL
  python main.py ask "вопрос"             — задать вопрос по загруженным документам
  python main.py bot                      — запустить Telegram-бота
  python main.py api get <url>            — тест GET-запроса к открытому API
"""

import sys
import traceback
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from rag_agent import RAGEngine
from document_loader import DocumentLoader, validate_url_security, DocumentLoaderError
from telegram_bot import run_bot
from api_client import APIClient, APIError, APIResponse, fetch_data, DEFAULT_TEST_API


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_step(step: int, title: str, status: str) -> None:
    icon = "OK" if status == "success" else "FAIL" if status == "error" else "INFO"
    print(f"  [{icon}] Шаг {step}: {title}")


def _mask_key(value: str, visible: int = 4) -> str:
    """Видоизменяет ключ для безопасного вывода."""
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "*" * (len(value) - visible)


def _run_diagnostic(config) -> None:
    """Запускает полную диагностику RAG-ассистента."""
    _print_header("RAG-АССИСТЕНТ: Диагностика")

    # --- Шаг 1: Конфигурация ---
    try:
        _print_step(1, "Загрузка конфигурации", "success")
        print(f"    Модель чата: {config.model_id}")
        print(f"    Модель эмбеддингов: {config.embedding_model}")
        print(f"    Индекс Pinecone: {config.pinecone_index_name}")
        print(f"    URL GenAPI (чат): {config.genapi_url}")
        print(f"    URL эмбеддингов: {config.genapi_embeddings_url}")
    except ValueError as exc:
        _print_step(1, "Загрузка конфигурации", "error")
        print(f"    Ошибка: {exc}")
        print("\n  Не указаны обязательные переменные окружения.")
        print("  Создайте файл .env и укажите ключи.")
        return
    except Exception as exc:
        _print_step(1, "Загрузка конфигурации", "error")
        print(f"    Ошибка: {exc}")
        return

    # --- Шаг 2: Подключение к Pinecone ---
    try:
        engine = RAGEngine(config)
        engine.connect_pinecone()
        _print_step(2, "Подключение к Pinecone", "success")
    except Exception as exc:
        _print_step(2, "Подключение к Pinecone", "error")
        print(f"    Ошибка: {exc}")
        traceback.print_exc()
        return

    # --- Шаг 3: Проверка индекса ---
    try:
        stats = engine.check_index()
        _print_step(3, "Проверка индекса", "success" if stats["exists"] else "info")
        print(f"    {stats['info']}")
        if stats["total_vectors"] == 0:
            print("    ⚠ Индекс пуст — поиск вернёт пустой результат.")
    except Exception as exc:
        _print_step(3, "Проверка индекса", "error")
        print(f"    Ошибка: {exc}")
        traceback.print_exc()
        return

    # --- Шаг 4: Создание эмбеддинга ---
    test_query = "Чем Lada Vesta отличается от Skoda Octavia и какие особенности важны для путешествий?"
    try:
        embedding = engine.create_embedding(test_query)
        _print_step(4, "Создание эмбеддинга", "success")
        print(f"    Размерность: {len(embedding)}")
        print(f"    Первые 5 значений: {embedding[:5]}")
    except Exception as exc:
        _print_step(4, "Создание эмбеддинга", "error")
        print(f"    Ошибка: {exc}")
        traceback.print_exc()
        return

    # --- Шаг 5: Поиск в Pinecone ---
    try:
        results = engine.search(test_query, top_k=5)
        _print_step(5, "Поиск в Pinecone", "success" if results.found else "info")
        print(f"    Найдено совпадений: {len(results.results)}")
        if results.found:
            for i, r in enumerate(results.results, 1):
                preview = r.text_preview[:150] or "(нет текста)"
                print(f"    [{i}] ID={r.id}  score={r.score:.4f}")
                print(f"        {preview}")
        else:
            print(f"    {results.message}")
            msg_lower = results.message.lower()
            if "10054" in msg_lower or "connection" in msg_lower or "разорвал" in msg_lower:
                print(
                    "\n    Подсказка: соединение с data-plane Pinecone разрывается.\n"
                    "    Если сеть блокирует прямые запросы к индексу, укажите\n"
                    "    PINECONE_PROXY_URL в файле .env (значение прокси)."
                )
    except Exception as exc:
        _print_step(5, "Поиск в Pinecone", "error")
        print(f"    Ошибка: {exc}")
        traceback.print_exc()
        return

    # --- Шаг 6: Генерация ответа (если есть контекст) ---
    if results.found and results.results:
        try:
            context = "\n\n".join(
                f"[{i}] {r.text_preview or '(нет текста)'}"
                for i, r in enumerate(results.results, 1)
            )
            answer = engine.generate_answer(test_query, context)
            _print_step(6, "Генерация ответа", "success")
            print(f"    Ответ:\n{answer}")
        except Exception as exc:
            _print_step(6, "Генерация ответа", "error")
            print(f"    Ошибка: {exc}")
            traceback.print_exc()

    # --- Итог ---
    _print_header("Итог")
    print(f"    Вопрос: {test_query}")
    print(f"    Совпадений: {len(results.results)}")
    print(f"    Статус: {'✓ Интеграция работает' if results.found else '✗ Результаты поиска отсутствуют'}")
    print(f"\n  Готово.")


def _run_load_urls(config, urls: list[str]) -> None:
    """Загружает URL в Pinecone."""
    _print_header("RAG-АССИСТЕНТ: Загрузка документов из URL")

    # --- Шаг 1: Подключение ---
    try:
        engine = RAGEngine(config)
        engine.connect_pinecone()
        _print_step(1, "Подключение к Pinecone", "success")
    except Exception as exc:
        _print_step(1, "Подключение к Pinecone", "error")
        print(f"    Ошибка: {exc}")
        return

    # --- Шаг 2: Создание загрузчика ---
    try:
        loader = DocumentLoader(
            embedding_client=engine._embedding_client,
            index=engine._index,
            chunk_size=500,
            chunk_overlap=50,
        )
        _print_step(2, "Загрузчик документов создан", "success")
    except Exception as exc:
        _print_step(2, "Загрузчик документов создан", "error")
        print(f"    Ошибка: {exc}")
        return

    # --- Шаг 3: Загрузка каждого URL ---
    all_success = 0
    all_failed = 0

    for idx, url in enumerate(urls, 1):
        print(f"\n  [{'=' * 56}]")
        print(f"  Документ {idx}/{len(urls)}: {url}")
        print(f"  {'-' * 56}")

        try:
            # Безопасная проверка URL
            validate_url_security(url)
            _print_step(3, "Проверка безопасности URL", "success")
            print(f"    URL корректен и указывает на публичный адрес.")
        except DocumentLoaderError as exc:
            _print_step(3, "Проверка безопасности URL", "error")
            print(f"    Ошибка: {exc}")
            all_failed += 1
            continue
        except Exception as exc:
            _print_step(3, "Проверка безопасности URL", "error")
            print(f"    Ошибка: {exc}")
            all_failed += 1
            continue

        # Загрузка и сохранение
        print(f"\n  Загрузка и обработка...")
        doc = loader.load_url(url)

        if doc.success:
            _print_step(4, f"Загрузка '{url}'", "success")
            print(f"    Заголовок: {doc.page_title or '(нет заголовка)'}")
            print(f"    Извлечено символов: {len(doc.full_text)}")
            print(f"    Фрагментов создано: {doc.total_chunks}")
            print(f"    Векторов сохранено: {doc.saved_vectors}")
            all_success += 1
        else:
            _print_step(4, f"Загрузка '{url}'", "error")
            print(f"    Ошибка: {doc.error_message}")
            all_failed += 1

    # --- Итог ---
    _print_header("Итог")
    print(f"    Загружено успешно: {all_success}/{len(urls)}")
    print(f"    Ошибок: {all_failed}/{len(urls)}")
    if all_success > 0:
        print(f"\n  Теперь можно задавать вопросы по загруженным документам.")
        print(f"  Запустите 'python main.py' для проверки поиска.")
    else:
        print(f"\n  Документы не загружены. Проверьте URL и сетевые настройки.")
    print(f"\n  Готово.")


def _run_ask(config, question: str) -> None:
    """Задаёт вопрос по загруженным документам."""
    _print_header("RAG-АССИСТЕНТ: Вопрос по документам")

    # --- Шаг 1: Подключение ---
    try:
        engine = RAGEngine(config)
        engine.connect_pinecone()
        _print_step(1, "Подключение к Pinecone", "success")
    except Exception as exc:
        _print_step(1, "Подключение к Pinecone", "error")
        print(f"    Ошибка: {exc}")
        return

    # --- Шаг 2: Проверка индекса ---
    try:
        stats = engine.check_index()
        _print_step(2, "Проверка индекса", "success" if stats["exists"] else "info")
        print(f"    {stats['info']}")
        if stats["total_vectors"] == 0:
            print("    ⚠ Индекс пуст — задайте вопрос или загрузите документы.")
    except Exception as exc:
        _print_step(2, "Проверка индекса", "error")
        print(f"    Ошибка: {exc}")
        return

    # --- Шаг 3: Поиск и генерация ответа ---
    print(f"\n  Вопрос: {question}")
    try:
        result = engine.ask(question, top_k=5)
        _print_step(3, "Поиск и генерация ответа", "success" if result.search.found else "info")

        if result.search.found and result.search.results:
            print(f"\n  Найдено фрагментов: {len(result.search.results)}")
            for i, r in enumerate(result.search.results, 1):
                preview = r.text_preview[:200] or "(нет текста)"
                score = r.score if r.score is not None else 0
                print(f"    [{i}] score={score:.4f}  id={r.id}")
                print(f"        {preview}")

            # Собираем уникальные источники
            sources = set()
            for r in result.search.results:
                url = r.metadata.get("source_url", "")
                if url:
                    sources.add(url)

            if sources:
                print(f"\n  Источники:")
                for src in sorted(sources):
                    print(f"    • {src}")

        else:
            print(f"\n  {result.search.message}")

        if result.answer:
            _print_step(4, "Ответ", "success")
            print(f"\n  Ответ:\n{result.answer}\n")
        else:
            _print_step(4, "Ответ", "info")
            print("\n  Ответ не сгенерирован.")

    except Exception as exc:
        _print_step(3, "Поиск и генерация ответа", "error")
        print(f"    Ошибка: {exc}")
        traceback.print_exc()
        return

    # --- Итог ---
    _print_header("Итог")
    print(f"    Вопрос: {question}")
    print(f"    Найдено фрагментов: {len(result.search.results)}")
    has_sources = any(
        r.metadata.get("source_url")
        for r in (result.search.results if result.search.found else [])
    )
    if result.search.found and result.answer:
        print(f"    Статус: ✓ Ответ сгенерирован")
        if has_sources:
            print(f"    Источники: указаны выше")
    else:
        print(f"    Статус: ✗ Ответ не найден")
    print(f"\n  Готово.")


def _run_api(args: list[str]) -> None:
    """Тест GET-запроса к открытому API.

    Команда: python main.py api get <URL>
    Команда: python main.py api test   — тестовый запрос к jsonplaceholder
    """
    _print_header("RAG-АССИСТЕНТ: Тест GET-запроса к открытому API")

    if not args:
        print("  Использование:")
        print("    python main.py api get <URL>")
        print("    python main.py api test   — тестовый запрос к jsonplaceholder")
        print()
        print(f"  Примеры:")
        print(f"    python main.py api get {DEFAULT_TEST_API}/posts/1")
        print(f"    python main.py api get {DEFAULT_TEST_API}/users?_limit=3")
        print(f"    python main.py api test")
        return

    command = args[0].lower()

    if command == "test":
        # Тестовый запрос к jsonplaceholder
        url = f"{DEFAULT_TEST_API}/posts/1"
        _print_step(1, f"Тестовый GET-запрос к {url}", "info")
    elif command == "get":
        if len(args) < 2:
            print("  Ошибка: укажите URL для GET-запроса")
            print("  Использование: python main.py api get <URL>")
            return
        url = args[1]
        _print_step(1, f"GET-запрос к {url}", "info")
    else:
        print(f"  Неизвестная команда: {command}")
        print("  Используйте: get <URL> или test")
        return

    # --- Выполняем запрос ---
    try:
        response = fetch_data(url)
    except APIError as exc:
        _print_step(2, "Результат запроса", "error")
        print(f"    {exc}")
        print(f"\n  Готово. Запрос не выполнен.")
        return
    except Exception as exc:
        _print_step(2, "Результат запроса", "error")
        print(f"    Неожиданная ошибка: {exc}")
        traceback.print_exc()
        print(f"\n  Готово. Запрос не выполнен.")
        return

    # --- Вывод результата ---
    _print_step(2, "Результат запроса", "success" if response.success else "error")
    print(f"    URL: {response.url}")
    print(f"    HTTP-код: {response.status_code}")
    print(f"    Успех: {'Да' if response.success else 'Нет'}")

    if response.success and response.data is not None:
        # Выводим первые 500 символов JSON
        data_str = str(response.data)[:500]
        print(f"\n    Данные ({len(str(response.data))} символов):")
        print(f"    {data_str}")
        if len(str(response.data)) > 500:
            print(f"    ... (показаны первые 500 из {len(str(response.data))} символов)")
    elif not response.success:
        print(f"    Ошибка: {response.error_message}")

    # --- Итог ---
    _print_header("Итог")
    print(f"    URL: {url}")
    print(f"    HTTP-код: {response.status_code}")
    if response.success:
        print(f"    Статус: ✓ GET-запрос выполнен успешно")
        print(f"    Ответ получен и распарсен (JSON)")
    else:
        print(f"    Статус: ✗ GET-запрос не выполнен")
    print(f"\n  Готово.")


def main() -> None:
    """Главная функция: диагностика, загрузка URL или вопрос."""
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        # Режим загрузки URL: python main.py load <url1> [url2] ...
        if len(sys.argv) < 3:
            print("Использование: python main.py load <URL> [URL2] ...")
            print("Пример: python main.py load https://example.com/article")
            sys.exit(1)

        urls = sys.argv[2:]
        try:
            config = load_config()
        except ValueError as exc:
            print(f"Ошибка конфигурации: {exc}")
            print("Создайте файл .env и укажите ключи.")
            sys.exit(1)
        except Exception as exc:
            print(f"Ошибка конфигурации: {exc}")
            sys.exit(1)

        _run_load_urls(config, urls)

    elif len(sys.argv) > 1 and sys.argv[1] == "ask":
        # Режим вопроса: python main.py ask "вопрос"
        if len(sys.argv) < 3:
            print("Использование: python main.py ask \"Ваш вопрос?\"")
            print("Пример: python main.py ask \"Кто создал Python?\"")
            sys.exit(1)

        question = " ".join(sys.argv[2:])
        try:
            config = load_config()
        except ValueError as exc:
            print(f"Ошибка конфигурации: {exc}")
            print("Создайте файл .env и укажите ключи.")
            sys.exit(1)
        except Exception as exc:
            print(f"Ошибка конфигурации: {exc}")
            sys.exit(1)

        _run_ask(config, question)

    elif len(sys.argv) > 1 and sys.argv[1] == "bot":
        # Режим Telegram-бота: python main.py bot
        try:
            config = load_config()
        except ValueError as exc:
            print(f"Ошибка конфигурации: {exc}")
            print("Создайте файл .env и укажите ключи.")
            sys.exit(1)
        except Exception as exc:
            print(f"Ошибка конфигурации: {exc}")
            sys.exit(1)

        if not config.telegram_bot_token:
            print("Ошибка: TELEGRAM_BOT_TOKEN не указан в .env")
            print("Получите токен у @BotFather в Telegram и добавьте в .env")
            print("Пример: TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz")
            sys.exit(1)

        run_bot(config)

    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        # Режим GET-запроса к открытому API: python main.py api get <url>
        _run_api(sys.argv[2:])

    else:
        # Режим диагностики (по умолчанию)
        try:
            config = load_config()
        except ValueError as exc:
            print(f"Ошибка конфигурации: {exc}")
            print("Создайте файл .env и укажите ключи.")
            sys.exit(1)
        except Exception as exc:
            print(f"Ошибка конфигурации: {exc}")
            sys.exit(1)

        _run_diagnostic(config)


if __name__ == "__main__":
    main()

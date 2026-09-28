# Зависимости

Команды установки выполняются из корня проекта: `python -m pip install -r requirements/<файл>`.
Файлы `.lock` закрепляют проверенные версии, `.txt` описывают диапазоны допустимых
версий или отдельный режим. Вложенные зависимости подключаются через `-r`.

| Файл | Назначение |
| --- | --- |
| `base.txt`, `base.lock` | Backend: HTTP, контракты и безопасное чтение XML |
| `ml.txt`, `ml.lock` | Численные расчёты и классическая ML-модель; включают base |
| `semantic.txt`, `semantic.lock` | ONNX-модели и токенизаторы; включают ml |
| `pilot.lock` | Полный научный анализ, SQLite, ключи и PDF; включает semantic |
| `web.txt` | Streamlit и веб-клиент для отдельного веб-окружения |
| `dev.txt`, `dev.lock` | Тесты; lock также включает Ruff и mypy |
| `native-build.lock` | Инструменты компиляции закреплённой SQLite |
| `audit.lock` | Аудит зависимостей в отдельном окружении |
| `cuda.txt` | Необязательный GPU runtime для Windows/Linux |

Перед установкой `pilot.lock` выполните `python -m scripts.build_sqlite_runtime`.
Он создаёт нужный wheel в `build/wheels/`. Это сборка зависимости приложения.

`cuda.txt` применяется к уже подготовленному окружению по инструкции
[локального AI](../docs/guides/local-ai.md); CPU- и GPU-пакеты ONNX Runtime
не должны быть установлены одновременно.

# Trendanalyser

Приложение для поиска научных публикаций и анализа технологических направлений.
В проекте есть настольный интерфейс, локальный веб-интерфейс, API, анализ слабых
сигналов, работа с несколькими источниками и сохранёнными результатами.

**[Полная документация проекта](docs/documentation.md)** — архитектура, стек,
установка на трёх ОС, HTTP API, методика анализа и сопровождение.

**Windows: [установка и запуск](#windows).**

Проект распространяется в виде исходного кода. Для работы нужны Python 3.13,
Tkinter и зависимости выбранного режима. Модели устанавливаются отдельно;
для поиска новых публикаций нужен интернет.

## Структура проекта

```text
app/                      Код приложения
├── backend/              Сбор публикаций, источники, хранилище и история
├── ml/                   Анализ корпуса и смысловая проверка трендов
├── pilot/                Научный анализ, доказательства, модели и отчёты
│   ├── approved_sources/ Адаптеры дополнительных источников
│   └── multisource/      Сопоставление сигналов из разных источников
├── radar/                Поиск и оценка технологических направлений
├── runtime/              Фоновые задания, модели, ключи и резервные копии
├── signal_model/         Модель слабых сигналов и её оценка
├── ui/                   Настольный и веб-интерфейсы, шрифты и стили
├── main.py               Запуск настольного приложения
└── web_*.py              API, администрирование и сервисы веб-интерфейса

launchers/                Файлы для запуска двойным щелчком
├── macos/                Настольная версия, локальный веб, веб-демо
└── windows/              Веб-демо, панель управления, настройка туннеля
requirements/             Зависимости по назначению и закреплённые версии
resources/models/         Реестр моделей и сведения об их лицензиях
scripts/                  Установка, запуск, оценка моделей и обслуживание
tools/                   Проверки, диагностика и измерение производительности
tests/                   Автоматические тесты и контрольные примеры
data/                    Входные материалы и сохранённые корпуса
docs/                    Документация по темам
.github/workflows/       Проверки исходников на Windows, macOS и Linux
pyproject.toml           Настройки тестов, проверки стиля и типов
```

`storage/` создаётся для локальных моделей и данных, `.venv/` и `.venv-web/` — для
Python-окружений, `build/` — для временных файлов SQLite и отчётов проверок.
Эти каталоги исключены из Git и не входят в передаваемые исходники.
Пользовательская библиотека настольного приложения по умолчанию хранится отдельно
от проекта, в системной папке профиля.

Каталоги `app/` разделены по функциям. `scripts/` содержит команды эксплуатации,
`tools/` — инструменты разработчика. Файлы Python названы в `snake_case`,
документы — в `kebab-case`. Идентификаторы профилей, форматы сохранённых данных и
версии методик сохранены для совместимости с существующими анализами.

<a id="windows"></a>

## Запуск на Windows

### 1. Подготовка

Установите **Python 3.13 x64 с Tcl/Tk**, **Git** и **Visual Studio Build Tools**
с компонентами MSVC C++ и Windows SDK. Компилятор нужен для SQLite.
Первичная установка зависимостей и моделей требует интернета;
локальная модель Qwen занимает около 1,8 ГБ, дополнительно нужны место под
остальные модели и рабочие данные.

Откройте **PowerShell**. Если проект ещё не скачан:

```powershell
git clone --branch main --single-branch https://github.com/fellichita/TEMA.git
cd TEMA
```

Если проект уже скачан, откройте PowerShell в его корневой папке — той, где
находятся `app/`, `requirements/` и `README.md`.

### 2. Установка и первый запуск

Выполняйте команды по очереди, дожидаясь успешного завершения каждой.
Если команда завершилась ошибкой, исправьте её перед следующим шагом.
Активация окружения и изменение политики выполнения PowerShell не требуются.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -c "import tkinter; print('Tk', tkinter.TkVersion)"
.\.venv\Scripts\python.exe -m pip install --upgrade pip==26.2.1
.\.venv\Scripts\python.exe -m pip install -r requirements/semantic.lock -r requirements/native-build.lock
.\.venv\Scripts\python.exe -m scripts.build_sqlite_runtime
.\.venv\Scripts\python.exe -m pip install -r requirements/pilot.lock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.setup_models
.\.venv\Scripts\python.exe -m scripts.install_local_llm
.\.venv\Scripts\python.exe -m app.main
```

Последняя команда открывает настольное приложение. Модели скачиваются при первой
подготовке; выбранный по умолчанию провайдер `local` использует установленный Qwen.
Поиск новых публикаций требует интернета.

### 3. Повторный запуск

После установки откройте PowerShell в папке проекта и выполните:

```powershell
.\.venv\Scripts\python.exe -m app.main
```

### Локальный веб-интерфейс на Windows

После подготовки основной `.venv` выполните следующие команды в PowerShell
из корня проекта. Если терминал занят настольным приложением, закройте его окно
или откройте второй терминал в той же папке.

```powershell
py -3.13 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements/web.txt
.\.venv-web\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.setup_models --web-analysis --profile storage/web-local-profile
.\.venv\Scripts\python.exe -m scripts.run_web_demo --local-only --no-auth --open-browser --data-dir storage/web-local-profile
```

Откроется браузер по адресу `http://127.0.0.1:8501`. Панель владельца доступна
на `http://127.0.0.1:8502`. Для повторного запуска достаточно последней команды;
для остановки нажмите **Ctrl+C** в терминале. Веб использует отдельный профиль
`storage/web-local-profile`.

Публичная ссылка и назначение файлов `launchers/windows/` описаны в
[руководстве по веб-развёртыванию](docs/guides/web-deployment.md).
Подробности моделей и диагностика — в [полной документации](docs/documentation.md).

## Быстрый запуск на macOS

На Apple Silicon с macOS 14+ установите Python 3.13 с Tkinter и Xcode Command Line
Tools. Затем откройте нужный файл:

- `launchers/macos/desktop.command` — настольное приложение;
- `launchers/macos/web-local.command` — локальный веб с подготовкой моделей.

При первом запуске устанавливаются зависимости. Подробности:
[подготовка macOS](docs/guides/macos-setup.md).

## Ручная подготовка окружения

Команды выполняются из корня проекта. Для SQLite нужен C-компилятор: Xcode
Command Line Tools на macOS, MSVC Build Tools на Windows или компилятор C на Linux.
Python должен содержать Tkinter.

```sh
python -m venv .venv
```

Активируйте окружение: `source .venv/bin/activate` на macOS/Linux или
`.venv\Scripts\Activate.ps1` в PowerShell на Windows. Затем:

```sh
python -m pip install --upgrade pip==26.2.1
python -m pip install -r requirements/semantic.lock -r requirements/native-build.lock
python -m scripts.build_sqlite_runtime
python -m pip install -r requirements/pilot.lock
python -m pip check
python -m scripts.setup_models
python -m app.main
```

`setup_models` готовит модели анализа и перевода. Для AI-провайдера `local`
дополнительно выполните `python -m scripts.install_local_llm` или воспользуйтесь
установкой модели в настройках приложения. При использовании внешнего провайдера
ключи задаются в приложении. Подробнее: [локальный AI](docs/guides/local-ai.md).

Веб-интерфейс использует отдельное окружение `.venv-web/` и
`requirements/web.txt`. Подготовка, запуск и файлы для Windows описаны в
[руководстве по веб-интерфейсу](docs/guides/web-deployment.md).

## Проверки

Для разработки дополнительно установите `requirements/dev.lock` и
`requirements/web.txt`. Браузерные проверки требуют Node.js 24.

```sh
python -m ruff check app scripts tools tests
python -m mypy app scripts tools/run_checks.py tests/_checks_plugin.py
```

Полный набор запускается через `tools.run_checks`: он выполняет GUI-тесты в
изолированных процессах. Команды для каждой ОС и диагностика находятся в
[руководстве по проверкам](docs/guides/checks.md).

## Данные и документация

- [Единая документация и пояснительная записка](docs/documentation.md).
- [Навигация по документации](docs/README.md).
- [Назначение файлов зависимостей](requirements/README.md).
- [Сохранённые корпуса](data/offline-storage/README.md) — исходные материалы для
  воспроизводимого анализа; восстановление: `python -m scripts.restore_storage`.
- `tests/fixtures/` — контрольные данные регрессионных тестов.

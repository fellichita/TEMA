# Воспроизводимые проверки

Запускайте команды из корня проекта в Python 3.13 с Tkinter и Node.js 24. Полный набор использует
локальные синтетические данные и сохранённые корпуса; API-ключи источников и загрузка
весов модели для тестов не требуются. Node.js должен быть доступен как `node`
или по абсолютному пути в `TREND_TEST_NODE`.

## Подготовка окружения

На macOS и Linux:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip==26.2.1
.venv/bin/python -m pip install -r requirements/dev.lock -r requirements/semantic.lock -r requirements/native-build.lock -r requirements/web.txt
.venv/bin/python -m scripts.build_sqlite_runtime
.venv/bin/python -m pip install -r requirements/pilot.lock
.venv/bin/python -m pip check
.venv/bin/python -m ruff check app scripts tools tests
.venv/bin/python -m mypy app scripts tools/run_checks.py tests/_checks_plugin.py
```

На Windows выполните те же шаги из PowerShell, заменив `.venv/bin/python` на
`.venv\Scripts\python.exe`; начальное окружение создаётся `py -3.13 -m venv .venv`.

Системный Python может содержать SQLite с известной ошибкой WAL. Сборщик
создаёт отдельный DB-API wheel `pysqlite3==0.6.0+sqlite3.53.4` из закреплённых
исходников SQLite и драйвера, проверяя SHA-256 до сборки и официальные SHA3-256
архива SQLite и `sqlite3.c`. Требуется SQLite 3.53.2 или новее: исправления WAL
в старых ветках не включают исправление FTS5 для CVE-2026-11822/11824.
Основание: [таблица SQLite CVE](https://sqlite.org/cves.html) и
[SQLite 3.53.4](https://sqlite.org/releaselog/3_53_4.html). Нужны штатные инструменты
C-компиляции (Xcode Command Line Tools на macOS, MSVC Build Tools на Windows).
Готовый публичный `pysqlite3==0.6.0` не подходит: проверенный macOS wheel содержит
SQLite 3.51.1. `--offline` использует ранее проверенные исходники в `build/runtime-sources`.
Приложение не запускает базу с затронутым runtime. Для диагностики:
`python -c "from app.sqlite_runtime import sqlite3; print(sqlite3.sqlite_version)"`.
Сборка SQLite не изменяет системные библиотеки.
Для отдельной сборки без установки используйте
`python -m scripts.build_sqlite_runtime --build-root build/runtime-security`.
Частная версия драйвера проверяется по исходникам и фактически загруженной SQLite
отдельно от аудита PyPI; её нельзя заменять публичной версией `0.6.0` в отчёте.

## Полный набор по платформам

### macOS

```bash
.venv/bin/python -m tools.run_checks --mode all --fail-on-skip \
  --exclude-test tests/test_run_web_demo.py::test_windows_children_keep_system_root_whatever_its_spelling \
  "Windows environment variable check is inapplicable on macOS"
```

На macOS GUI-тесты требуют интерактивной графической сессии. Runner сначала
собирает все тесты, запускает non-GUI проверки одним процессом, проверяет доступность
Tk и затем последовательно выполняет каждый GUI-тест в отдельном процессе. Это
избегает проблем с повторными Tcl/Tk interpreters в общем процессе. Запуск всего
набора обычным `pytest tests` не обеспечивает эту изоляцию.

### Windows

Запускайте из PowerShell в обычной графической сессии. Исключаются только десять
проверок POSIX, неприменимых к Windows; список совпадает с CI.

```powershell
.\.venv\Scripts\python.exe -m tools.run_checks --mode all --fail-on-skip `
  --exclude-test tests/test_runtime_backup.py::test_importing_local_fifo_is_rejected_without_blocking "POSIX mkfifo has no Windows equivalent" `
  --exclude-test tests/test_runtime_jobs.py::test_checkpoint_replaced_by_fifo_cannot_block_result_or_shutdown "POSIX mkfifo has no Windows equivalent" `
  --exclude-test tests/test_pilot_service_integration.py::test_local_result_fifo_is_rejected_without_blocking_ui_executor "POSIX mkfifo has no Windows equivalent" `
  --exclude-test tests/test_pilot_profiles.py::test_directory_sync_failure_after_commit_keeps_new_selection_with_warning "Directory fsync is POSIX-specific" `
  --exclude-test tests/test_pilot_profiles.py::test_restore_postcommit_sync_failure_returns_new_backend_and_explicit_warning "Directory fsync is POSIX-specific" `
  --exclude-test tests/backend/test_repository.py::test_new_database_is_private "Windows uses profile ACLs rather than POSIX file modes" `
  --exclude-test tests/test_backend_lock_safety.py::test_lock_closes_existing_permissive_profile_and_file "Windows uses profile ACLs rather than POSIX file modes" `
  --exclude-test tests/test_pilot_sources.py::test_archive_revisions_are_private_even_in_an_existing_permissive_directory "Windows uses profile ACLs rather than POSIX file modes" `
  --exclude-test tests/test_runtime_backup.py::test_package_is_private_while_compression_is_writing "Windows uses profile ACLs rather than POSIX file modes" `
  --exclude-test tests/test_runtime_jobs.py::test_analysis_database_and_checkpoints_are_private_on_disk "Windows uses profile ACLs rather than POSIX file modes"
```

### Linux

Установите Node.js 24 отдельно и проверьте `node --version`: он должен вывести
`v24.x.x`. Системный пакет `nodejs` может содержать другую версию. Для GUI без
дисплея используйте Xvfb:

```bash
sudo apt-get install xvfb xauth openbox wmctrl libtk8.6 libfontconfig1
PATH="$PWD/.venv/bin:$PATH" bash scripts/check_linux.sh
```

Скрипт исключает единственный Windows-тест окружения. Этот же скрипт запускает CI: Xvfb с DPI 96, Openbox и проверка готовности WM.
Для изолированного локального Linux-воспроизведения доступен Docker:

```bash
docker build -f tools/Dockerfile.checks -t trendanalyser-checks .
docker run --init --rm trendanalyser-checks --mode gui tests/ui/test_visible_pages.py
```

Контейнер использует Debian; итоговая проверка Ubuntu выполняется в GitHub Actions.
Пользовательские библиотеки, ключи, окружение и дистрибутивы в build context не входят.
Образ задаёт `PYTHONDONTWRITEBYTECODE=1` и отдельный
`PYTHONPYCACHEPREFIX=/tmp/trendanalyser-checks-pycache`: при bind mount Python
не читает `__pycache__` с хоста. Один запрет записи `.pyc` не запрещает их чтение.
Для проверки в ранее собранном образе добавьте к `docker run` параметры
`--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPYCACHEPREFIX=/tmp/trendanalyser-checks-pycache`.
Каталог находится внутри нового контейнера и не монтируется с хоста; существующие
пользовательские кеши не удаляются. Это устраняет перенос host bytecode и его
`co_filename`; само наличие host-пути в traceback ещё не доказывает устаревший код.

Отдельная проверка отзывчивости выполняет изменение размера уже показанной
таблицы из 50 синтетических документов в пяти новых процессах. Запускайте её
при свободном CPU после полной регрессии:

```bash
.venv/bin/python -m tools.measure_resize --samples 5 --output build/checks/resize-performance.json
# Linux: тот же Xvfb/Openbox, что и в основной регрессии
PATH="$PWD/.venv/bin:$PATH" bash scripts/check_linux.sh --resize-performance --samples 5 --output build/checks/resize-performance.json
```

Каждый замер должен уложиться в 2 секунды wall time, 1 секунду CPU и 1 секунду
задержки event loop, без callback errors и с фактически изменившейся геометрией.
Поздний успех функционального predicate не отменяет превышение бюджета.
Отчёт содержит платформу, отдельные версии Tcl/Tk, hash исходников и все замеры.
Это проверка source runtime при масштабе 100%, а не cold startup, real-corpus
benchmark или измерение установленного пакета. GUI-геометрия 150/200% проверяется
основным набором отдельно.

Проверяйте `python -c 'import tkinter'` именно у выбранного interpreter. Установка
системного `python3-tk` не добавляет Tk в другой Python, собранный без `_tkinter`.
Runner поддерживает Linux, macOS и Windows. На Windows тест запускается после
назначения Job Object; закрытие родителя, timeout и завершение проверки закрывают
дерево дочерних процессов. Windows workflow использует Windows Server 2025,
запускает применимые тесты исходников с `--fail-on-skip`.
Как и Linux, Windows workflow обязательно проверяет установленные distributions
основного Python через отдельное окружение `build/audit-tools` из
`requirements/audit.lock`; это включает зависимости только для Windows.
`scripts.audit_runtime_dependencies` запускается Python приложения и передаёт
изолированному аудитору полный список публичных версий с `--strict`. Пропуск,
неполный результат или неизвестная частная версия завершают шаг с ошибкой.
Для закреплённого частного SQLite отдельно обязательны проверенные исходники,
совпадение загруженного `sqlite_source_id()`, файлов установленного драйвера
с собранным wheel и работа FTS5. Итог `audit-summary.json`, публичный результат
`dependencies.json` и `private-sqlite-proof.json` сохраняются в
`build/checks/security` и включаются в общий CI artifact. Для локального запуска:
`python -m scripts.audit_runtime_dependencies --auditor-python build/audit-tools/bin/python`.
На Windows путь аудитора — `build/audit-tools/Scripts/python.exe`. Отдельная сборка
SQLite задаётся параметром `--sqlite-build-root build/runtime-security`.
Только перечисленные POSIX-проверки исключены по точным node ID с причиной;
`summary.json` сохраняет полный размер коллекции и `excluded_tests`.
Этот CI не подтверждает установку, подпись или полную приёмку на Windows 11.

## Адресные проверки

Выбирайте конкретные файлы или node ID. При запуске полного режима `all` или
`non-gui` добавляйте применимые исключения из раздела своей платформы:

```bash
.venv/bin/python -m tools.run_checks --mode collect
.venv/bin/python -m tools.run_checks --mode gui
.venv/bin/python -m tools.run_checks --mode non-gui tests/test_check_runner.py
.venv/bin/python -m tools.run_checks --mode gui tests/ui/test_operation_state.py
```

Без optional semantic runtime допустим локальный прогон с явными ONNX skips:
установите `requirements/dev.lock` и `requirements/ml.lock`, запустите без
`--fail-on-skip`. Такой результат не подтверждает native semantic runtime.
В полном CI установлен semantic lock и включён `--fail-on-skip`: skipped, xfailed
и xpassed не считаются полной успешной приёмкой.

Каждый запуск создаёт новый `build/checks/<run-id>/` с `summary.json`, manifest,
JSON каждого pytest-процесса и его stdout/stderr. Журнал `.events.jsonl` немедленно
сохраняет начало/окончание теста и результаты фаз; он остаётся после аварии процесса.
Перед deadline отдельный `.stacks.log` получает стеки потоков. Summary содержит
точный commit, dirty flag, Python/OS и версии установленных distributions, а также
уникальные failed node IDs с учётом subtests. Можно указать другой родительский
каталог через `--output-dir`. Старые отчёты не переиспользуются. Summary отдельно
показывает ошибки, предупреждения, пропуски и `not_completed` — тесты без полного
завершения. Код 0 возвращается только при успешном выполнении запланированного
набора; timeout, сигнал, ошибка сборки тестов, отсутствие отчёта или несовпадение
набора возвращают ненулевой код. Режим `collect` проверяет только сбор тестов.

Лимиты по умолчанию: 120 секунд на collection, 900 секунд на non-GUI процесс,
60 секунд на каждый GUI-тест. Последние два настраиваются `--suite-timeout` и
`--gui-timeout`. По timeout или прерыванию останавливается группа дочернего
процесса; промежуточный отчёт сохраняется. Не увеличивайте timeout, чтобы скрыть
зависание: сначала изучите соответствующий `.log`.

Внутри `TkCase` функциональное ожидание состояния и mapping имеет бюджет 10 секунд;
явные короткие deadlines отмены и отдельных операций сохранены. Это не норматив
скорости приложения: startup SLO проверяется отдельно через `scripts.measure_startup`
на целевой ОС. В Linux воспроизведении с CPU-квотой 0,5 готовность тестового
приложения наблюдалась через 7,62 секунды; прежние 3 секунды проверяли скорость
dispatch вместе с корректностью. Условия видимости, размеров, фокуса и отсутствия
обрезания не ослаблены. `ui_waits` в JSON call-report и `.events.jsonl` хранит
исходную строку predicate, время первого успеха, возврата event loop, CPU и бюджет.
Эти времена остаются видны и при успешном тесте. Записи ограничены 256 ожиданиями;
тексты полей и значения замыканий не записываются. Session events сохраняют доступные
Linux `cpu.max`/`cpu.stat`, чтобы отличать CPU throttling от ожидания приложения.

GUI-тесты имеют зарегистрированный marker `gui`. Базовый `TkCase` передаёт его
всем наследникам, включая классы за пределами `tests/ui`. Отдельный прямой Tk-класс
или тест следует пометить `@pytest.mark.gui`. Guard в non-GUI процессе блокирует
немаркированное создание графического Tk до обращения к дисплею и делает весь
прогон неуспешным, даже если тест перехватывает ошибку. Безоконный `Tk(useTk=False)`
допустим. Не создавайте root при импорте тестового модуля.

Runner отключает автозагрузку внешних pytest plugins и не наследует `PYTEST_ADDOPTS`
или `PYTEST_PLUGINS`; настройки репозитория и выбранные пути применяются явно.
Это исключает случайную фильтрацию или изменение результата локальными настройками.
Собственные проверки runner используют временные мини-наборы pytest и не создают
настоящих окон.

Workflow `.github/workflows/checks.yml` использует Linux/Xvfb и тот же runner.
Версии Actions v7 сверены с официальными примерами
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python) и
[upload-artifact](https://github.com/actions/upload-artifact).
Наличие workflow само по себе не означает, что Linux-приёмка уже состоялась:
её результат определяется реальным завершённым CI-прогоном и его артефактами.

Windows workflow дополнительно запускает `python -m tools.profile_gui_startup
--output build/checks/startup-profile.json` перед обязательным набором тестов.
Это диагностика двух свежих процессов с отображённым/скрытым окном и небольшим
in-memory backend. Она сохраняет время/CPU, ограниченные таблицы профилирования
главного потока и длительность backend-вызовов. Её отдельный `continue-on-error`
позволяет получить полный тестовый отчёт даже при неудачном профилировании;
обязательный test step и его десятисекундные функциональные ожидания не меняются.
Время с профилировщиком не является performance acceptance, а внутренние native
кадры Tk требуют отдельного native sampling.

`.gitattributes` фиксирует LF для текстовых файлов, чтобы protected-source hashes
не зависели от checkout Windows. Эти правила входят в `source_tree_sha256`.
В Docker с bind-mounted исходниками всегда передавайте **обе** переменные:
`PYTHONDONTWRITEBYTECODE=1` и `PYTHONPYCACHEPREFIX=/tmp/trendanalyser-checks-pycache`.
Первая запрещает запись `.pyc`, но сама по себе не запрещает чтение старого
host-bytecode; отдельный cache prefix исключает такую подмену проверяемого кода.

## Диагностика внутренних сбоев

`app.diagnostics` использует стандартный logger без изменения глобальной
конфигурации. После открытия backend безопасные события ERROR записываются в
`logs/diagnostics.log` активной библиотеки. Лимит файла — 256 КиБ, сохраняются
три предыдущих файла; на POSIX каталог закрыт режимом 0700, файлы — 0600.
При переключении библиотеки прежний файловый обработчик закрывается.
До открытия библиотеки или при недоступности её лога стандартный logger может
вывести безопасную запись в stderr.

Запись содержит фиксированный код события, тип исключения и ограниченный стек
с именем функции и строкой. Текст исключения, цепочка причин, исходные строки,
локальные переменные, запросы и полные пути не включаются. Ожидаемые ошибки
параметров, backend и отмена обрабатываются обычным пользовательским сообщением.
Ошибка самого обработчика логирования не заменяет исходную ошибку операции.
Файловый обработчик принимает только записи `app.diagnostics`, созданные его
собственными функциями; сторонние logger и произвольные сообщения туда не пишут.
Внутренний сбой CPU-процесса получает диагностический ID и проверенную родителем
категорию/точку вызова. Логи не включаются ни в публичный результат, ни в backup.

Mypy gate не является полной строгой типизацией всего исторического кода:
тела неаннотированных функций дополнительно проверяются в backend, runtime,
pilot, всех UI-модулях, diagnostics и runner/plugin. Для старых
неаннотированных ML-функций сохраняются обычные ограничения mypy;
информационные `annotation-unchecked` notes не скрываются. Ruff gate ограничен
`E9,F,B`; стилевые долги не устранялись массовым форматированием.

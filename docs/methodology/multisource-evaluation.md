# Проверка полезности многоканальных сигналов

Этот протокол существует **отдельно от приложения**. Файл `tests/evaluation/multisource-v1.json` задаёт три направления, T0/T1/T2, семь веток и правила разметки; он не содержит результаты, человеческие оценки или выгрузки источников. Настольное приложение остаётся на научном стартовом экране до завершения независимых R1-проверок и продуктового gate. Ни запуск тестов, ни успешный импорт не доказывают качество отбора технологий.

## Подготовить реальные срезы

1. Предметный специалист фиксирует для каждой из трёх областей узкий `scope_query`, дату и своё имя **до** сбора. Исходные предложенные формулировки в protocol JSON не являются подтверждёнными задачами.
2. Для каждого среза заранее фиксируются UTC `collection_started_at`, затем UTC `knowledge_cutoff`, hash действующей policy, caps расходов для всех семи веток и неизменяемая карта `source_id → concept family`. Не переносить остаток бюджета одного источника другому. Для T1/T2 выдержать минимум 14 дней от предыдущего cutoff.
3. Сохранить реальные завершённые `.trendresult` для C и `.trendsignals` для B/S-вариантов в одном каталоге с capture JSON. Пакеты проходят полную проверку состава, архивных ссылок и разрешений на передачу. Экспорт `local_only` не допускается; для таких данных нужен разрешённый extract/право или отдельная локальная схема оценки, которую этот протокол пока не заявляет как переносимую.
4. Не выполнявшуюся или упавшую ветку записать как `not_executed`/`failed` с причиной, нулевыми расходами и без artifact/hash. Она остаётся в знаменателе. Дату/результат задним числом не подменять.
5. End-to-end ветки самостоятельно собирают кандидатов со своими источниками; термины из отключённого канала туда не попадают. Для fixed-pool отдельно сохранить вывод всех семи веток на одном объявленном наборе семей, полученном из TOP-5 end-to-end без B плюс пяти заранее воспроизводимо выбранных контролей на область. В fixed-pool внешние вызовы и расходы должны быть нулевыми.

Capture лежит по умолчанию в `build/multisource/captures/T0.json` (позже `T1.json`, `T2.json`). Его строгая схема в `CutCapture`, `CaseCapture`, `ArmCapture` в `scripts/evaluate_multisource.py`. Каждый case содержит `area_id`, `scope_query`, `scope_confirmed_by`, `scope_frozen_at`, `budget_plan_frozen_at`, `budget_caps_micro` для всех arms, `family_map`, `control_population`, `fixed_pool_families`, а также семь записей в `end_to_end` и семь в `fixed_pool` строго в порядке `C, B, S, S+Q, S+A, S+Q+A, S+Q+A+F`. Успешная запись содержит относительное имя пакета, SHA-256 файла, проверяемый `root_hash`, `input_hash`, расход в микроединицах и число внешних вызовов. `suite_hash` получается из протокола и проверяется автоматически. Схема отказывает при неизвестных полях, будущих alias/источниках, несовпадении policy, утечке выключенного канала и неразрешённом бюджете.

## Воспроизведение без сети

Из корня репозитория:

```bash
.venv/bin/python -m scripts.evaluate_multisource inventory --suite tests/evaluation/multisource-v1.json
.venv/bin/python -m scripts.evaluate_multisource replay --suite tests/evaluation/multisource-v1.json --cut T0 --mode end-to-end --offline --output build/multisource/T0
.venv/bin/python -m scripts.evaluate_multisource replay --suite tests/evaluation/multisource-v1.json --cut T0 --mode fixed-pool --offline --output build/multisource/T0/fixed-pool
```

`inventory` сообщает только о наличии capture, не об истинности его содержания. `replay` повторно читает каждый пакет и сверяет транзитивные ссылки, policy, scope, UTC границу, права, выбранную candidate universe и SHA-256. Он не запускает сеть, ML или получение свежего Wordstat. `manifest.json` содержит незаслеплённые позиции для вычисления метрик; его не отправляют разметчикам.

## Слепая независимая R1-разметка

```bash
.venv/bin/python -m scripts.evaluate_multisource review-packets --manifest build/multisource/T0/manifest.json --output build/multisource/T0/review
```

Команда вновь воспроизводит манифест и выдаёт `review/packets.json` и пустой `review/labels.csv`. Для concept family создаётся один нейтральный пакет с доступными до cutoff данными, независимо от числа веток. Для каждого отличающегося показываемого payload создаётся отдельное задание с его собственными утверждениями и ссылками. B не является дополнительной discovery-разметкой. В reviewer packet отсутствуют arm, rank и итоговый балл; источник и ограничения свидетельств видны. Порядок фиксирован хешем среза. До меток выбирается не менее 25% заданий для второго рецензента со стратификацией по области и типу задания. Первичных заданий не больше 195 на срез и 650 по протоколу.

Разметчик заполняет уже подготовленные строки `primary`/`secondary` CSV. В каждое заполненное решение входят собственный `reviewer_id`, UTC `label_available_at`, объяснение, refs и значения `yes/no/unknown` по применимым полям. Для concept это пять пунктов rubric и `concept_useful_now`; для payload — `factual_claims_supported`, `next_step_useful`, `payload_supported` и `critical_unsupported`. Второй человек не видит первичную запись до своего решения. При расхождении добавить строку `adjudication` с третьим отличающимся `reviewer_id`. Техническая проверка ловит повторные ID/фазы, пропуски и несовместимый словарь; личность и независимость людей остаются организационной ответственностью. Записи `unknown` не засчитываются положительно.

```bash
.venv/bin/python -m scripts.evaluate_multisource score --manifest build/multisource/T0/manifest.json --labels build/multisource/T0/review/labels.csv --output build/multisource/T0/quality.json
```

Оценка заново проверяет исходный replay и пакеты. `useful_slots_at_5` имеет фиксированный знаменатель пять: пустое место равно пустому месту; неизвестная метка даёт интервал `[known_yes/5, (known_yes+unknown_returned)/5]`; неисполненный case — `[0,1]`. Суммарный отчёт сохраняет все три направления и семь веток. Поддержка каждого payload, полезность следующей проверки и критически неподтверждённые утверждения отдельны от метки концепта. `review_complete` значит только, что предусмотренная разметка структурно завершена; это **не** автоматическое право поменять стартовый экран. Для gate ещё нужны T1 и закрытый T2, предметный разбор денежных/научных ошибок и сравнение времени аналитика. Поле `default_screen_change_eligible` остаётся `false` до отдельного решения с реальными свидетельствами.

Для fixed-pool `score` использует тот же `labels.csv` и нейтральные concept tasks end-to-end. Новые fixed-pool payloads без идентичного хеша остаются неразмеченными в отчёте и не наследуют оценку другой ветки.

## Закрытый T2 и отдельный L2

Перед T2 завершить T1 review, сохранить `quality.json`, `manifest.json`, `review/packets.json` и `review/labels.csv` в одной директории среза. `T2.json` ссылается на относительный T1 report через `t1_review_file`, содержит SHA-256 этого report, `t1_review_completed_at` и `policy_frozen_at` после завершения разметки, но до начала T2. Evaluator проверяет T1 report повторным replay и CSV, а не верит одному `status`. Для команд T2 нужен явный `--allow-blind`; результаты T2 не используют для подбора policy.

L2 не смешивается с R1 CSV. После фактического `knowledge_cutoff + 180 дней` можно передать `score --outcomes outcomes.csv` с колонками `area_id,family_id,independent_confirmation_180d,reviewer_id,label_available_at,independent_source_ref`. До горизонта все families имеют статус `censored`, отсутствие строки после него — `unknown`, а не `no`. Это не метрика прибыли и не результат собственного score. Официальные права на выгрузки, реальное покрытие источников и независимость экспертов нельзя доказать unit-тестом.

На 20 сентября 2026 года `inventory` показывает отсутствующие T0/T1/T2 captures. Реальных R1 labels и созревших L2 outcomes нет; статус качества — `not_evaluated/pending`. Синтетические данные в unit-тестах проверяют только отказы и арифметику evaluator.

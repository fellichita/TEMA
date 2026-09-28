# Источники backend

Сбор работает из консоли или Python API без интерфейса и ML.
Crossref и OpenAlex проверены реальными запросами. EPO OPS реализован и покрыт
тестовыми HTTP/XML-ответами, но живой доступ требует ключей и пока не проверен.

| ID | Данные | Доступ в этой реализации |
| --- | --- | --- |
| crossref | Научные публикации, DOI, метаданные | Без ключа |
| openalex | Научные публикации, DOI/OpenAlex ID, аннотации при наличии | Без ключа для базовых запросов; необязательный OPENALEX_API_KEY |
| epo | Патентные публикации, библиография, семейство | Обязательны EPO_OPS_KEY и EPO_OPS_SECRET |

## Команды

Из корня проекта, PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.backend sources
.\.venv\Scripts\python.exe -m app.backend --data-dir storage collect "neuromorphic computing" --source openalex --limit 20 --from-date 2024-01-01
.\.venv\Scripts\python.exe -m app.backend --data-dir storage collect "neuromorphic computing" --sources crossref openalex --limit 20 --from-date 2024-01-01
.\.venv\Scripts\python.exe -m app.backend --data-dir storage documents --limit 20
.\.venv\Scripts\python.exe -m app.backend --data-dir storage versions "doi:10.1038/s41586-024-08253-8"
```

Последняя команда сработает, только если такой DOI уже собран; иначе будет
`document_not_found`. Ключ нужного документа берите из `documents` → `document_key`.

В Python:

```python
from pathlib import Path
from app.backend.config import BackendSettings
from app.backend.contracts import SearchRequest
from app.backend.service import Backend

with Backend(BackendSettings(data_dir=Path("storage"))) as backend:
    jobs = backend.collect_many(
        SearchRequest(topic="neuromorphic computing", max_results=100),
        sources=("crossref", "openalex"),
    )
    for job in jobs:
        print(job.request.source, job.state, job.stored, job.error_code)
    documents = backend.list_documents(limit=100)
```

Лимит применяется к каждому источнику отдельно. Задания идут последовательно через
один worker, с отдельными статусами, отменой и снимками. Ошибка EPO не мешает
запланированному после него OpenAlex. Частичный успех возвращает код CLI 1,
но уже собранные документы сохраняются. `coverage_complete=false` означает,
что нельзя считать выборку полным корпусом для статистики роста.

## Доступ к EPO

1. Зарегистрируйте доступ к [EPO OPS](https://www.epo.org/en/searching-for-patents/data/web-services/ops)
   и получите consumer key и consumer secret для своего приложения.
2. Задайте `EPO_OPS_KEY` и `EPO_OPS_SECRET` в окружении запускаемого процесса.
   В PyCharm это Environment variables конфигурации запуска. Не включайте значения
   в общую конфигурацию проекта, исходники, чат, README или Git.
3. В том же окружении выполните `python -m app.backend sources`:
   `credentials_configured=true` проверяет только наличие переменных, не их действительность.
4. Выполните:

```powershell
.\.venv\Scripts\python.exe -m app.backend --data-dir storage collect "neuromorphic computing" --source epo --limit 10 --from-date 2024-01-01
```

Ключи не записываются backend в базу, URL или журналы. OAuth-токен хранится в памяти
до закрытия адаптера. При отказе авторизации выполняется не более одного обновления
токена на поисковую страницу. Нет автоматического чтения `.env` и пока нет интеграции
с Windows Credential Manager. Переменные окружения — промежуточный механизм для
разработки, а не законченное решение хранения секретов коммерческого приложения.

## Данные и ограничения

- Одинаковый нормализованный DOI связывает публикации двух источников. Все версии
  доступны отдельно. При отсутствии DOI используется ID источника, без нечёткого слияния.
- Последняя версия в общей выдаче может иметь меньше полей, чем предыдущая:
  автоматическое слияние аннотаций/авторов пока не выполняется. Исходные версии не теряются.
- OpenAlex восстанавливает аннотацию из `abstract_inverted_index`; отсутствующий текст
  не выдумывается. Невалидная запись учитывается как `skipped`.
- EPO сохраняет исходную XML-библиографию. В `authors` попадают изобретатели;
  заявители доступны в XML, отдельного поля компаний пока нет.
- Патентная идентичность — страна + номер + код вида. A1 и B1 не сливаются.
  Семейство сохраняется для последующей группировки, а не служит ключом дедупликации.
- Для EPO используется дата публикации, не дата подачи или приоритета.
- EPO ограничен 100 записями на страницу и 2000 на запрос в этой реализации.
  При большем лимите и остатке выдачи `source_exhausted=false`. Разбиение большого
  запроса по периодам пока не автоматизировано.
- Для EPO пока не поддерживаются кавычки и обратная косая черта внутри темы.
  HTTP 404 рассматривается как ошибка, а не как доказательство пустой выдачи.
- Ответы ограничены по размеру и времени; редиректы запрещены, TLS проверяется,
  XML DTD и внешние сущности запрещены. Произвольные PDF и полный текст не скачиваются.
- Поисковая релевантность задаётся источником; результаты не являются готовым ТОП-15
  трендов. Источники могут по-разному трактовать одну тему и иметь разный охват.

Официальные спецификации:
[OpenAlex: авторизация](https://help.openalex.org/api/authentication/),
[OpenAlex: пагинация](https://help.openalex.org/api/paging/),
[OpenAlex: поля публикаций](https://help.openalex.org/data/works/attributes/),
[EPO OPS API](https://ops.epo.org/wsdl/ops.yaml).

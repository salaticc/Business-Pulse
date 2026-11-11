# G-Analytics · Business Pulse

[![CI](https://github.com/salaticc/Business-Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/salaticc/Business-Pulse/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.6.0-blue)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Учебный ML Engineering проект: прогноз дневной выручки, обнаружение
существенных отклонений и оценка правил предупреждений по стоимости ошибок.
Цель — определить, какой показатель требует проверки, сохраняя различие
между ошибкой данных, аномалией и подтверждённым инцидентом.

**Текущая версия: 0.6.0 — Alert Intelligence.** Код и эксперименты локальные.
Telegram-бот, Backend API и мобильное приложение не интегрированы.
Падение выручки не доказывает убыток; коммерческая эффективность не подтверждена.

## Быстрый запуск

Проверенная локальная среда: Windows x64, Python 3.14.2. Точные версии
библиотек — `requirements-lock.txt`. CI настроен на Python 3.14 для Windows
и Linux; результат текущего запуска показывает badge выше.

```powershell
git clone https://github.com/salaticc/Business-Pulse.git
cd Business-Pulse
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q

# Воспроизвести синтетическую лабораторию
.\.venv\Scripts\python.exe -m scripts.make_incident_demo

# Выбрать порог только на validation; TEST зарезервирован
.\.venv\Scripts\python.exe -m scripts.evaluate_alerts --validation-only --output outputs/v06_validation

# Проверить зафиксированный выбор на историческом TEST
.\.venv\Scripts\python.exe -m scripts.evaluate_alerts --output outputs/v06
Start-Process outputs/v06/alert_report.html
```

На Linux: `python3.14 -m venv .venv`, затем `.venv/bin/python` вместо
`.\.venv\Scripts\python.exe`. Активация окружения не обязательна.
Повторение опубликованного TEST — проверка воспроизводимости, не новый
независимый эксперимент. Новые правила нельзя подбирать по его результатам.

## Что реализовано

- Прогноз на следующий день: календарь, исторические лаги, seasonal naive,
  медиана аналогичных дней недели, HistGradientBoostingRegressor.
- Expanding-origin CV, выбор без доступа к финальному TEST; MAE, RMSE,
  WAPE и отчёт о разбросе по временным окнам.
- Alert Intelligence: абсолютное И относительное отклонение, проверка
  полноты/актуальности по метаданным, приоритеты, эпизоды и дедупликация.
- Синтетическая разметка, дневные TP/FP/FN/TN/Precision/Recall/F1,
  Recall эпизодов, индивидуальные цены пропусков и стоимость ложных сообщений.
- Выбор из 12 политик по validation, журнал причин и подтверждения
  учебной истиной, воспроизводимые CSV/JSON/HTML.
- 104 теста ML/alert-логики; CI, контроль состава и версий.

## Результат учебного эксперимента v0.6

Выбранная политика: падение **не менее 25% И 250 единиц выручки**.
Выбор сделан только на validation. Стоимости — условные предпочтения,
не реальные деньги и не предотвращённый ущерб.

| Показатель | Validation | TEST |
|---|---:|---:|
| Найдено размеченных эпизодов | 7 из 8 | 7 из 8 |
| Ложных локальных business warnings | 5 | 3 |
| Цена ложных сообщений | 15.00 | 9.00 |
| Цена пропущенных эпизодов | 18.75 | 15.00 |
| Общая условная стоимость | 33.75 | 24.00 |

Дневной Recall после подавления отличается от Recall целых эпизодов.
Полная confusion matrix и ограничения: [отчёт v0.6](REPORT_V06.md),
[числовой снимок](reports/v0.6.0/alert_report.json),
[все пороги на validation](reports/v0.6.0/threshold_comparison.csv).

## Контракт времени и данные

Прогноз разрешён после закрытия предыдущего торгового дня и подтверждения
полноты загрузки, до открытия прогнозируемого дня. Факты дня t не являются
признаками прогноза t. v0.6 проверяет явные отметки доступности; реальный
источник и календарь пока не подключены. Истинность метаданных невозможно
доказать по агрегированному CSV.

В Git включены только синтетические продажи и инциденты. Полный реальный
UCI Online Retail XLSX не включён; его можно положить в игнорируемый
`data/raw/`. В UCI нет разметки бизнес-проблем. Новая лаборатория не
превращает раскрытый TEST v0.4 в независимую проверку.

[Контракт](FORECAST_CONTRACT.md) ·
[Методология Alert Intelligence](ALERT_INTELLIGENCE.md) ·
[Источник реального набора](data/raw/README.md).

## Структура

```text
src/                     прогнозирование, загрузка, alert-правила и оценка
scripts/                 команды экспериментов и проверки репозитория
tests/                   корректность и воспроизводимость
configs/                 политики и модель затрат
data/demo/               синтетические продажи v0.4/v0.5
data/synthetic_incidents/ синтетическая лаборатория v0.6
lessons/                 исходные учебные материалы
reports/v0.4.0/           результаты MVP
reports/v0.5.0/           результаты ML Evaluation
reports/v0.6.0/           результаты Alert Intelligence
verification/            журналы выполненных локальных проверок
docs/history/            происхождение восстановленных этапов
outputs/                 новые рабочие результаты, игнорируются Git
```

Учебные задания сохранены как материалы для автора, а не поручение
автоматически выполнять следующие этапы.

## Совместимость с v0.5

Существующие модули сохранены. Старые команды доступны отдельно:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --validation-only --output outputs/v05_validation
.\.venv\Scripts\python.exe -m src.pipeline --input data/demo/daily_sales.csv
.\.venv\Scripts\python.exe -m scripts.compare_models --input data/demo/daily_sales.csv
.\.venv\Scripts\python.exe -m scripts.notify
```

`scripts.notify` по умолчанию показывает preview старого отфильтрованного
alerts.csv после запуска pipeline. Он не подключён к журналу v0.6.
Реальную доставку не включать без отдельного явного разрешения.

## Разработка и история

[CHANGELOG](CHANGELOG.md) · [Правила разработки](CONTRIBUTING.md) ·
[Происхождение архивов](docs/history/archive_provenance.json) ·
[Отчёт о переносе](docs/REPOSITORY_SETUP.md).

Этапы восстановлены из ZIP отдельными коммитами с текущими датами;
исходная история разработки не выдумывается. `VERSION` — версия приложения;
теги `v0.4.0`, `v0.5.0`, `v0.6.0` обозначают проверенные этапы.
Следующие изменения: ветка → тесты и эксперимент → PR → main →
новый неизменяемый тег. Лицензия: [MIT](LICENSE), Gleb Adamokov.

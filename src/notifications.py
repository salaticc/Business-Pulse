"""Telegram adapter for reviewed daily revenue alerts.

No bot token, network request, or delivery is performed during import.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen


def load_alerts(path: str | Path) -> list[dict[str, str]]:
    """Read only rows explicitly flagged by the pipeline as alerts."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("date")]


def format_alert(row: dict[str, str], *, demo: bool = True) -> str:
    """Prepare a human-readable message without claiming to know the root cause."""
    prefix = "🧪 УЧЕБНЫЕ ДАННЫЕ\n" if demo else ""
    date = row["date"]
    observed = float(row["revenue"])
    forecast = float(row["predicted"])
    drop_pct = float(row["drop_pct"])
    orders = int(float(row.get("orders") or 0))
    return (
        f"{prefix}⚠️ Business Pulse · отклонение продаж\n"
        f"Дата: {date}\n"
        f"Факт: {observed:,.2f}\n"
        f"Прогноз: {forecast:,.2f}\n"
        f"Ниже прогноза: {drop_pct:.1f}%\n"
        f"Заказов: {orders}\n\n"
        "Что проверить: полноту выгрузки, календарь работы, "
        "количество заказов и средний чек.\n"
        "Важно: это сигнал для проверки, а не доказанная причина потерь."
    )


def send_message(message: str, *, token: str, chat_id: str, opener: Callable = urlopen) -> None:
    """Send a message using the official HTTPS Telegram Bot API."""
    if not token or not chat_id:
        raise ValueError("Не заданы TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
    payload = json.dumps({"chat_id": chat_id, "text": message}, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # opener is injectable for isolated tests (no internet required).
    with opener(req, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError("Telegram API вернул ok=false")


def load_sent_dates(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    content = json.loads(p.read_text(encoding="utf-8"))
    return set(content.get("sent_dates", []))


def save_sent_dates(path: str | Path, dates: set[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps({"sent_dates": sorted(dates)}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(p)


def dispatch(
    rows: list[dict[str, str]],
    *,
    date: str | None,
    send: bool = False,
    demo: bool = True,
    state_path: str | Path = "local_state/sent_alerts.json",
    transport: Callable[..., None] = send_message,
) -> list[str]:
    """Preview by default; send only for an explicitly chosen date and with credentials."""
    if send and not date:
        raise ValueError("Для отправки необходимо явно передать --date YYYY-MM-DD")
    if not demo and not send:
        raise ValueError("Режим реальных данных предусмотрен для подтверждённой отправки")
    selected = [row for row in rows if not date or row["date"] == date]
    if not date:
        selected = selected[-3:]  # safe preview of only the last three historical alerts
    sent_dates = load_sent_dates(state_path) if send else set()
    previews = []
    for row in selected:
        text = format_alert(row, demo=demo)
        if send and row["date"] not in sent_dates:
            transport(
                text,
                token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            )
            sent_dates.add(row["date"])
            save_sent_dates(state_path, sent_dates)
        previews.append(text)
    return previews

"""Preview alerts, or explicitly send ONE chosen historical/demo alert.

Examples:
    python -m scripts.notify
    python -m scripts.notify --date 2025-08-20
    python -m scripts.notify --date 2025-08-20 --send
"""
import argparse
from src.notifications import dispatch, load_alerts


def main():
    parser = argparse.ArgumentParser(description="Business Pulse: alerts → Telegram adapter")
    parser.add_argument("--alerts", default="outputs/alerts.csv", help="CSV from src.pipeline")
    parser.add_argument("--date", help="Date YYYY-MM-DD; required for actual sending")
    parser.add_argument("--send", action="store_true", help="Actually send the selected alert (opt-in)")
    parser.add_argument("--state", default="local_state/sent_alerts.json")
    args = parser.parse_args()
    rows = load_alerts(args.alerts)
    previews = dispatch(rows, date=args.date, send=args.send, demo=True, state_path=args.state)
    if not previews:
        print("Алертов на выбранную дату нет. Ничего не отправлено.")
        return
    if not args.send:
        print("РЕЖИМ ПРЕДПРОСМОТРА: сообщения не отправлены.\n")
    for message in previews:
        print(message)
        print("=" * 50)
    if args.send:
        print("Отправка выполнена (повторная отправка за ту же дату блокируется).")


if __name__ == "__main__":
    main()

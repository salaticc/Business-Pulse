import json
import pytest
from src.notifications import dispatch, format_alert, load_alerts, load_sent_dates


def row(date="2025-08-20"):
    return {"date": date, "revenue": "90", "predicted": "150", "drop_pct": "40", "orders": "4"}


def test_format_honest_about_uncertainty():
    msg = format_alert(row())
    assert "УЧЕБНЫЕ ДАННЫЕ" in msg
    assert "40.0%" in msg
    assert "не доказанная причина" in msg


def test_dry_run_never_calls_network(tmp_path):
    def fail_transport(*args, **kwargs):
        raise AssertionError("Network must not be called during preview")
    previews = dispatch([row()], date=None, send=False, state_path=tmp_path / "state.json", transport=fail_transport)
    assert len(previews) == 1
    assert not (tmp_path / "state.json").exists()


def test_send_requires_explicit_date():
    with pytest.raises(ValueError):
        dispatch([row()], date=None, send=True)


def test_send_once_and_remember(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_ONLY")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    messages = []
    def fake_send(message, *, token, chat_id):
        messages.append((message, token, chat_id))
    p = tmp_path / "state.json"
    dispatch([row()], date="2025-08-20", send=True, state_path=p, transport=fake_send)
    dispatch([row()], date="2025-08-20", send=True, state_path=p, transport=fake_send)
    assert len(messages) == 1
    assert load_sent_dates(p) == {"2025-08-20"}


def test_filter_by_date(tmp_path):
    p = tmp_path / "alerts.csv"
    p.write_text("date,revenue,predicted,drop_pct,orders\n2025-01-01,5,10,50,1\n", encoding="utf-8")
    assert len(load_alerts(p)) == 1
    assert dispatch(load_alerts(p), date="2025-01-02") == []

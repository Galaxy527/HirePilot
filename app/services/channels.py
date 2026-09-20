"""Outbound email / SMS channels (SMTP + Twilio, with log/outbox fallback)."""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from app.config import Config
from app.services import metrics as metrics_svc

logger = logging.getLogger(__name__)


def _write_outbox(kind: str, to: str, subject: str, body: str) -> Path:
    Config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = Config.OUTBOX_DIR / f"{kind}_{ts}.txt"
    path.write_text(
        f"to: {to}\nsubject: {subject}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def send_email(to: str, *, subject: str, body: str) -> bool:
    to = (to or "").strip()
    if not to:
        return False
    full_body = body
    if Config.MAIL_LOG_ONLY or not Config.MAIL_ENABLED:
        path = _write_outbox("email", to, subject, full_body)
        logger.info("Email outbox -> %s (%s)", path.name, to)
        metrics_svc.incr("notify_email_outbox")
        return True

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = Config.MAIL_DEFAULT_SENDER
    msg["To"] = to
    msg.set_content(full_body)
    try:
        with smtplib.SMTP(Config.MAIL_SERVER, Config.MAIL_PORT, timeout=20) as smtp:
            if Config.MAIL_USE_TLS:
                smtp.starttls()
            if Config.MAIL_USERNAME:
                smtp.login(Config.MAIL_USERNAME, Config.MAIL_PASSWORD)
            smtp.send_message(msg)
        metrics_svc.incr("notify_email_sent")
        return True
    except Exception as exc:
        logger.warning("SMTP send failed, writing outbox: %s", exc)
        _write_outbox("email_failed", to, subject, f"{full_body}\n\nERROR: {exc}")
        metrics_svc.incr("notify_email_failed")
        return False


def send_sms(to: str, *, body: str) -> bool:
    to = (to or "").strip()
    if not to:
        return False
    text = (body or "")[:500]
    if (
        Config.SMS_LOG_ONLY
        or not Config.SMS_ENABLED
        or Config.SMS_PROVIDER == "log"
        or not (Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN and Config.TWILIO_FROM_NUMBER)
    ):
        path = _write_outbox("sms", to, "SMS", text)
        logger.info("SMS outbox -> %s (%s)", path.name, to)
        metrics_svc.incr("notify_sms_outbox")
        return True

    try:
        from twilio.rest import Client

        client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        client.messages.create(to=to, from_=Config.TWILIO_FROM_NUMBER, body=text)
        metrics_svc.incr("notify_sms_sent")
        return True
    except Exception as exc:
        logger.warning("SMS send failed, writing outbox: %s", exc)
        _write_outbox("sms_failed", to, "SMS", f"{text}\n\nERROR: {exc}")
        metrics_svc.incr("notify_sms_failed")
        return False

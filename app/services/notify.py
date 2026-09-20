"""In-app + email/SMS notifications."""
from __future__ import annotations

import logging

from app.config import Config
from app.models import Notification, User, db
from app.services.channels import send_email, send_sms

logger = logging.getLogger(__name__)


def _absolute_link(link: str) -> str:
    link = (link or "").strip()
    if not link:
        return Config.APP_PUBLIC_URL
    if link.startswith("http://") or link.startswith("https://"):
        return link
    return f"{Config.APP_PUBLIC_URL}{link if link.startswith('/') else '/' + link}"


def _dispatch_channels(user: User, *, title: str, body: str, link: str = "") -> None:
    abs_link = _absolute_link(link)
    text = f"{title}\n{body}\n{abs_link}".strip()
    if getattr(user, "notify_email", True) and (user.email or "").strip():
        send_email(user.email, subject=f"[HirePilot] {title}", body=text)
    if getattr(user, "notify_sms", False) and (user.phone or "").strip():
        send_sms(user.phone, body=f"[HirePilot] {title}: {body}"[:200])


def notify_user(
    user_id: int,
    *,
    title: str,
    body: str = "",
    link: str = "",
) -> Notification:
    n = Notification(user_id=user_id, title=title, body=body, link=link, is_read=False)
    db.session.add(n)
    db.session.commit()
    user = db.session.get(User, user_id)
    if user:
        try:
            _dispatch_channels(user, title=title, body=body, link=link)
        except Exception:
            logger.exception("channel notify failed for user %s", user_id)
    return n


def notify_hrs(*, title: str, body: str = "", link: str = "", org_id: int | None = None) -> int:
    q = User.query.filter_by(role="hr")
    if org_id is not None:
        q = q.filter_by(org_id=org_id)
    hrs = q.all()
    for hr in hrs:
        db.session.add(
            Notification(user_id=hr.id, title=title, body=body, link=link, is_read=False)
        )
    db.session.commit()
    for hr in hrs:
        try:
            _dispatch_channels(hr, title=title, body=body, link=link)
        except Exception:
            logger.exception("channel notify failed for hr %s", hr.id)
    return len(hrs)


def unread_count(user_id: int) -> int:
    return Notification.query.filter_by(user_id=user_id, is_read=False).count()


def list_notifications(user_id: int, limit: int = 30) -> list[Notification]:
    return (
        Notification.query.filter_by(user_id=user_id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
        .all()
    )


def mark_read(user_id: int, notification_id: int | None = None) -> int:
    q = Notification.query.filter_by(user_id=user_id, is_read=False)
    if notification_id is not None:
        q = q.filter_by(id=notification_id)
    rows = q.all()
    for n in rows:
        n.is_read = True
    db.session.commit()
    return len(rows)

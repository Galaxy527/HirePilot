"""Append-only audit trail."""
from __future__ import annotations

import json
import logging

from flask import has_request_context, request
from flask_login import current_user

from app.models import AuditLog, db

logger = logging.getLogger(__name__)


def audit(
    action: str,
    entity_type: str,
    entity_id: str | int | None = None,
    detail: dict | None = None,
    actor_id: int | None = None,
    org_id: int | None = None,
) -> AuditLog:
    if actor_id is None and has_request_context():
        try:
            if current_user.is_authenticated:
                actor_id = int(current_user.id)
                if org_id is None:
                    org_id = getattr(current_user, "org_id", None)
        except Exception:
            actor_id = None
    ip = None
    if has_request_context():
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    row = AuditLog(
        org_id=org_id,
        actor_id=actor_id,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        detail_json=json.dumps(detail or {}, ensure_ascii=False),
        ip=ip,
    )
    db.session.add(row)
    db.session.commit()
    logger.info("audit action=%s entity=%s:%s actor=%s", action, entity_type, entity_id, actor_id)
    return row

"""Tenant / organization helpers."""
from __future__ import annotations

from app.config import Config
from app.models import Organization, db


def get_or_create_default_org() -> Organization:
    code = Config.DEFAULT_ORG_CODE
    org = Organization.query.filter_by(code=code).first()
    if org:
        return org
    org = Organization(name=Config.DEFAULT_ORG_NAME, code=code)
    db.session.add(org)
    db.session.commit()
    return org


def resolve_org_by_code(code: str | None) -> Organization | None:
    c = (code or "").strip().upper()
    if not c:
        return get_or_create_default_org()
    return Organization.query.filter_by(code=c).first()

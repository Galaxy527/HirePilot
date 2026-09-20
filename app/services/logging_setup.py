"""Request-scoped logging with request_id / user_id."""
from __future__ import annotations

import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, g, has_request_context, request
from flask_login import current_user


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if has_request_context():
            record.request_id = getattr(g, "request_id", "-")
            try:
                record.user_id = (
                    current_user.id if current_user.is_authenticated else "-"
                )
            except Exception:
                record.user_id = "-"
        else:
            record.request_id = "-"
            record.user_id = "-"
        return True


def setup_logging(app: Flask, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] req=%(request_id)s user=%(user_id)s "
        "%(name)s: %(message)s"
    )
    ctx = ContextFilter()

    file_handler = RotatingFileHandler(
        log_dir / "hirepilot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(ctx)
    file_handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(ctx)
    console.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(file_handler)
        root.addHandler(console)

    @app.before_request
    def _bind_request_id():
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        app.logger.info("%s %s", request.method, request.path)

    @app.after_request
    def _attach_request_id(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response

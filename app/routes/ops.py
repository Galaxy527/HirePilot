"""Ops / SRE console — not part of HR product UI."""
from __future__ import annotations

import logging
from functools import wraps

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from app.config import Config
from app.services.audit import audit

logger = logging.getLogger(__name__)
bp = Blueprint("ops", __name__, url_prefix="/ops")


def ops_required(fn):
    """Only HR admins (运维/管理员) may access ops console."""

    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_hr or not current_user.is_hr_admin:
            abort(403)
        return fn(*args, **kwargs)

    return wrapper


@bp.route("/")
@ops_required
def home():
    return render_template("ops/home.html")


@bp.route("/rag-eval")
@ops_required
def rag_eval():
    """Visualize latest RAG / RAGAS enterprise evaluation reports."""
    import json

    from app.config import BASE_DIR

    reports_dir = BASE_DIR / "evals" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(reports_dir.glob("rag_eval_*.json"), reverse=True)
    selected = (request.args.get("f") or "").strip()
    report_path = None
    if selected:
        cand = reports_dir / selected
        if (
            cand.is_file()
            and cand.name.startswith("rag_eval_")
            and cand.suffix == ".json"
        ):
            report_path = cand
    if report_path is None and files:
        report_path = files[0]

    report = None
    error = ""
    if report_path is None:
        error = "暂无评测报告。请点击「重新评测」或运行：python scripts/eval_rag.py --enterprise"
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            error = f"无法读取报告：{exc}"

    file_list = [
        {"name": p.name, "mtime": p.stat().st_mtime, "size": p.stat().st_size}
        for p in files[:20]
    ]
    return render_template(
        "ops/rag_eval.html",
        report=report,
        error=error,
        report_name=report_path.name if report_path else "",
        files=file_list,
    )


@bp.route("/rag-eval/run", methods=["POST"])
@ops_required
def rag_eval_run():
    """Trigger enterprise RAG eval. Blocks until done (~1 min)."""
    import subprocess
    import sys

    from app.config import BASE_DIR

    try:
        proc = subprocess.run(
            [sys.executable, "scripts/eval_rag.py", "--enterprise"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            flash("企业级 RAG 评测完成（等级见看板）", "success")
        else:
            flash(
                f"评测未全部通过（exit={proc.returncode}），请打开看板查看失败门禁",
                "error",
            )
        audit(
            "rag_eval_run",
            "eval",
            detail={"exit": proc.returncode, "tail": (proc.stdout or "")[-400:]},
        )
    except subprocess.TimeoutExpired:
        flash("评测超时（>5min），请稍后查看 reports 目录", "error")
    except Exception as exc:
        logger.exception("rag eval run failed")
        flash(f"评测失败：{exc}", "error")
    return redirect(url_for("ops.rag_eval"))


@bp.route("/logs")
@ops_required
def backend_logs():
    """Tail hirepilot.log for ops debugging."""
    from pathlib import Path

    log_path = Path(Config.LOG_DIR) / "hirepilot.log"
    lines: list[str] = []
    error = ""
    try:
        if not log_path.exists():
            error = f"日志文件不存在：{log_path}"
        else:
            raw = log_path.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines()[-400:]
    except Exception as exc:
        logger.exception("read logs failed")
        error = str(exc)
    return render_template(
        "ops/logs.html",
        lines=lines,
        error=error,
        log_path=str(log_path),
        line_count=len(lines),
    )

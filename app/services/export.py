"""CSV / PDF export helpers."""
from __future__ import annotations

import csv
import io
from datetime import datetime


def resumes_to_csv(resumes: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "resume_id",
            "candidate",
            "job",
            "total_score",
            "recommendation",
            "pipeline_stage",
            "meets_threshold",
            "assigned_hr",
            "created_at",
        ]
    )
    for r in resumes:
        w.writerow(
            [
                r.id,
                r.candidate.display_name if r.candidate else "",
                r.target_job_title or "",
                r.total_score if r.total_score is not None else "",
                r.recommendation or "",
                r.pipeline_stage or "",
                "yes" if r.meets_threshold else "no",
                r.assigned_hr.display_name if r.assigned_hr else "",
                r.created_at.isoformat() if r.created_at else "",
            ]
        )
    return buf.getvalue()


def audits_to_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "actor_id", "action", "entity_type", "entity_id", "detail", "ip", "created_at"])
    for a in rows:
        w.writerow(
            [
                a.id,
                a.actor_id or "",
                a.action,
                a.entity_type,
                a.entity_id or "",
                a.detail_json or "",
                a.ip or "",
                a.created_at.isoformat() if a.created_at else "",
            ]
        )
    return buf.getvalue()


def resumes_to_pdf_bytes(resumes: list, *, title: str = "候选人导出") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 18
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()}Z  count={len(resumes)}")
    y -= 24
    c.setFont("Helvetica", 9)
    for r in resumes:
        line = (
            f"#{r.id} {r.candidate.display_name if r.candidate else '?'} | "
            f"{r.target_job_title or '-'} | score={r.total_score} | "
            f"{r.recommendation or '-'} | stage={r.pipeline_stage}"
        )
        # reportlab default fonts lack CJK — transliterate display for PDF safety
        safe = line.encode("ascii", "replace").decode("ascii")
        if y < 50:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 9)
        c.drawString(40, y, safe[:110])
        y -= 14
    c.save()
    return buf.getvalue()


def audits_to_pdf_bytes(rows: list, *, title: str = "Audit Export") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 20
    c.setFont("Helvetica", 8)
    for a in rows:
        line = f"{a.id} {a.created_at} {a.action} {a.entity_type}:{a.entity_id} actor={a.actor_id}"
        safe = line.encode("ascii", "replace").decode("ascii")
        if y < 50:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 8)
        c.drawString(40, y, safe[:120])
        y -= 12
    c.save()
    return buf.getvalue()

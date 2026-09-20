"""HR-only routes."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import wraps

from flask import (
    Blueprint,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from app.config import Config
from app.extensions import limiter
from app.models import (
    AuditLog,
    ChatSession,
    JobPosting,
    LiveInterview,
    Resume,
    ResumeNote,
    User,
    db,
)
from app.services.access import can_force_search, hr_resume_query, require_hr_resume
from app.services.audit import audit
from app.services.chat_context import hr_context_note
from app.services.export import audits_to_csv, audits_to_pdf_bytes, resumes_to_csv, resumes_to_pdf_bytes
from app.services.metrics import incr
from app.services.notify import notify_user
from app.services.scoring import build_details_for_resume
from app.services.weights import (
    get_dim_weights,
    get_org,
    get_rank_weights,
    parse_percent,
    recompute_org_totals,
    save_weights,
)

logger = logging.getLogger(__name__)
bp = Blueprint("hr", __name__, url_prefix="/hr")

STAGE_LABELS = {
    "new": "新建",
    "screening": "筛选中",
    "interview": "面试中",
    "offer": "Offer",
    "hired": "已录用",
    "rejected": "已淘汰",
}
REJECT_LABELS = {
    "skill_mismatch": "技能不匹配",
    "experience_short": "经验不足",
    "culture_fit": "文化匹配",
    "salary": "薪资预期",
    "duplicate": "重复投递",
    "other": "其他",
}


def hr_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_hr:
            abort(403)
        return fn(*args, **kwargs)

    return wrapper


def _rank_key(resume: Resume, resume_w: float, interview_w: float) -> float:
    return resume.composite_score(resume_w, interview_w)


@bp.route("/")
@hr_required
def dashboard():
    q = (request.args.get("q") or "").strip()
    force = bool(q)
    resume_w, interview_w = get_rank_weights(current_user.org_id)

    if force:
        if not can_force_search(current_user):
            flash("强制查询仅限 HR 管理员", "error")
            return redirect(url_for("hr.dashboard"))
        resumes = _force_search(q)
        audit("hr_force_search", "search", detail={"q": q, "hits": len(resumes)})
    else:
        resumes = (
            hr_resume_query(current_user)
            .filter(Resume.meets_threshold.is_(True), Resume.processing_status == "ready")
            .order_by(Resume.total_score.desc())
            .all()
        )

    resumes = sorted(resumes, key=lambda r: _rank_key(r, resume_w, interview_w), reverse=True)
    return render_template(
        "hr/dashboard.html",
        resumes=resumes,
        threshold=Config.RESUME_SCORE_THRESHOLD,
        query=q,
        force_mode=force,
        resume_w=resume_w,
        interview_w=interview_w,
        stage_labels=STAGE_LABELS,
        is_admin=current_user.is_hr_admin,
    )


def _force_search(q: str) -> list[Resume]:
    base = hr_resume_query(current_user)
    results: list[Resume] = []
    if q.isdigit():
        by_id = Resume.query.get(int(q))
        if by_id and by_id.org_id == current_user.org_id:
            results.append(by_id)
        user = User.query.get(int(q))
        if user and user.is_candidate and user.org_id == current_user.org_id:
            results.extend(user.resumes.all())

    users = User.query.filter(
        User.role == "candidate",
        User.org_id == current_user.org_id,
        User.display_name.contains(q) | User.username.contains(q),
    ).all()
    for u in users:
        results.extend(u.resumes.all())

    more = base.filter(Resume.raw_text.contains(q)).limit(50).all()
    results.extend(more)

    seen = set()
    unique = []
    for r in results:
        if r.id not in seen and r.org_id == current_user.org_id:
            seen.add(r.id)
            unique.append(r)
    return unique


@bp.route("/resume/<int:resume_id>/delete", methods=["POST"])
@hr_required
def delete_resume(resume_id: int):
    from app.services.pii import hr_delete_resume

    resume = Resume.query.get_or_404(resume_id)
    require_hr_resume(current_user, resume)
    ok = hr_delete_resume(
        resume_id, actor_id=current_user.id, org_id=current_user.org_id
    )
    if not ok:
        flash("简历不存在或无权删除", "error")
        return redirect(url_for("hr.dashboard"))
    flash(f"已删除简历 #{resume_id}", "success")
    incr("resume_deleted")
    return redirect(url_for("hr.dashboard"))


@bp.route("/resume/<int:resume_id>", methods=["GET", "POST"])
@hr_required
def resume_detail(resume_id: int):
    resume = Resume.query.get_or_404(resume_id)
    require_hr_resume(current_user, resume)

    if request.method == "POST":
        action = (request.form.get("action") or "").strip()
        if action == "pipeline":
            stage = (request.form.get("pipeline_stage") or "").strip()
            reject_reason = (request.form.get("reject_reason") or "").strip()
            reject_note = (request.form.get("reject_note") or "").strip()
            if stage not in Config.PIPELINE_STAGES:
                flash("无效的管道阶段", "error")
            elif stage == "rejected" and not reject_reason:
                flash("淘汰必须选择原因", "error")
            else:
                old = resume.pipeline_stage
                resume.pipeline_stage = stage
                if not resume.assigned_hr_id:
                    resume.assigned_hr_id = current_user.id
                if stage == "rejected":
                    resume.reject_reason = reject_reason
                    resume.reject_note = reject_note
                else:
                    resume.reject_reason = None
                    resume.reject_note = None
                db.session.commit()
                audit(
                    "pipeline_updated",
                    "resume",
                    resume.id,
                    detail={"from": old, "to": stage, "reject_reason": reject_reason},
                )
                notify_user(
                    resume.candidate_id,
                    title="申请状态更新",
                    body=f"简历 #{resume.id} 阶段变更为「{STAGE_LABELS.get(stage, stage)}」"
                    + (
                        f"（原因：{REJECT_LABELS.get(reject_reason, reject_reason)}）"
                        if stage == "rejected"
                        else ""
                    ),
                    link=f"/candidate/resume/{resume.id}",
                )
                flash("管道状态已更新", "success")
            return redirect(url_for("hr.resume_detail", resume_id=resume.id))

        if action == "assign":
            if not current_user.is_hr_admin:
                flash("仅管理员可改分派", "error")
            else:
                aid = request.form.get("assigned_hr_id", type=int)
                resume.assigned_hr_id = aid
                db.session.commit()
                audit("resume_assigned", "resume", resume.id, detail={"assigned_hr_id": aid})
                flash("已更新负责人", "success")
            return redirect(url_for("hr.resume_detail", resume_id=resume.id))

        if action == "claim":
            if resume.assigned_hr_id and resume.assigned_hr_id != current_user.id:
                flash("该简历已分派给其他招聘同学", "error")
            else:
                resume.assigned_hr_id = current_user.id
                db.session.commit()
                audit("resume_claimed", "resume", resume.id)
                flash("已认领该候选人", "success")
            return redirect(url_for("hr.resume_detail", resume_id=resume.id))

        if action == "note":
            body = (request.form.get("note_body") or "").strip()
            if not body:
                flash("备注不能为空", "error")
            else:
                db.session.add(
                    ResumeNote(resume_id=resume.id, author_id=current_user.id, body=body)
                )
                db.session.commit()
                audit("resume_note_added", "resume", resume.id, detail={"len": len(body)})
                flash("备注已添加", "success")
            return redirect(url_for("hr.resume_detail", resume_id=resume.id))

        if action == "schedule":
            return _schedule_live(resume)

    interviews = resume.interviews.all()
    resume_w, interview_w = get_rank_weights(current_user.org_id)
    composite = _rank_key(resume, resume_w, interview_w)
    score_report = build_details_for_resume(resume) if resume.processing_status == "ready" else {}
    if resume.processing_status == "ready" and not resume.score_details() and score_report:
        resume.set_score_details(score_report)
        db.session.commit()

    audit("hr_view_resume", "resume", resume.id)
    incr("hr_resume_views")
    notes = resume.notes.limit(50).all()
    live = resume.live_interviews.order_by(LiveInterview.starts_at.desc()).all()
    hrs = User.query.filter_by(role="hr", org_id=current_user.org_id).all()
    return render_template(
        "hr/resume_detail.html",
        resume=resume,
        interviews=interviews,
        live_interviews=live,
        composite=composite,
        threshold=Config.RESUME_SCORE_THRESHOLD,
        score_report=score_report,
        notes=notes,
        stages=Config.PIPELINE_STAGES,
        stage_labels=STAGE_LABELS,
        reject_reasons=Config.REJECT_REASONS,
        reject_labels=REJECT_LABELS,
        hrs=hrs,
        is_admin=current_user.is_hr_admin,
    )


def _schedule_live(resume: Resume):
    title = (request.form.get("title") or "真人面试").strip()
    starts = (request.form.get("starts_at") or "").strip()
    ends = (request.form.get("ends_at") or "").strip()
    location = (request.form.get("location") or "").strip()
    meeting_url = (request.form.get("meeting_url") or "").strip()
    notes = (request.form.get("live_notes") or "").strip()
    interviewer_id = request.form.get("interviewer_id", type=int) or current_user.id
    try:
        starts_at = datetime.fromisoformat(starts)
        ends_at = datetime.fromisoformat(ends)
        if starts_at.tzinfo is None:
            starts_at = starts_at.replace(tzinfo=timezone.utc)
        if ends_at.tzinfo is None:
            ends_at = ends_at.replace(tzinfo=timezone.utc)
    except ValueError:
        flash("请填写有效的开始/结束时间", "error")
        return redirect(url_for("hr.resume_detail", resume_id=resume.id))
    if ends_at <= starts_at:
        flash("结束时间须晚于开始时间", "error")
        return redirect(url_for("hr.resume_detail", resume_id=resume.id))

    live = LiveInterview(
        org_id=resume.org_id or current_user.org_id,
        resume_id=resume.id,
        scheduled_by=current_user.id,
        interviewer_id=interviewer_id,
        title=title,
        starts_at=starts_at,
        ends_at=ends_at,
        location=location,
        meeting_url=meeting_url,
        notes=notes,
        status="scheduled",
    )
    db.session.add(live)
    if resume.pipeline_stage in ("new", "screening"):
        resume.pipeline_stage = "interview"
    if not resume.assigned_hr_id:
        resume.assigned_hr_id = current_user.id
    db.session.commit()
    audit("live_interview_scheduled", "live_interview", live.id, detail={"resume_id": resume.id})
    notify_user(
        resume.candidate_id,
        title="真人面试已安排",
        body=f"{title} · {starts_at.strftime('%Y-%m-%d %H:%M')} — {ends_at.strftime('%H:%M')}",
        link="/candidate/calendar",
    )
    flash("真人面试已排期", "success")
    return redirect(url_for("hr.resume_detail", resume_id=resume.id))


@bp.route("/calendar")
@hr_required
def calendar():
    items = (
        LiveInterview.query.filter_by(org_id=current_user.org_id)
        .order_by(LiveInterview.starts_at.asc())
        .limit(200)
        .all()
    )
    if not current_user.is_hr_admin:
        items = [
            x
            for x in items
            if x.interviewer_id == current_user.id
            or x.scheduled_by == current_user.id
            or (x.resume and x.resume.assigned_hr_id in (None, current_user.id))
        ]
    return render_template("hr/calendar.html", items=items)


@bp.route("/calendar/<int:live_id>/cancel", methods=["POST"])
@hr_required
def calendar_cancel(live_id: int):
    live = LiveInterview.query.get_or_404(live_id)
    if live.org_id != current_user.org_id:
        abort(403)
    live.status = "cancelled"
    db.session.commit()
    audit("live_interview_cancelled", "live_interview", live.id)
    flash("已取消该场真人面试", "success")
    return redirect(url_for("hr.calendar"))


@bp.route("/export/resumes")
@hr_required
def export_resumes():
    fmt = (request.args.get("format") or "csv").lower()
    resumes = (
        hr_resume_query(current_user)
        .filter(Resume.processing_status == "ready")
        .order_by(Resume.total_score.desc())
        .all()
    )
    audit("export_resumes", "export", detail={"format": fmt, "count": len(resumes)})
    incr("exports")
    if fmt == "pdf":
        data = resumes_to_pdf_bytes(resumes, title="HirePilot Candidates")
        return Response(
            data,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=candidates.pdf"},
        )
    csv_data = resumes_to_csv(resumes)
    return Response(
        "\ufeff" + csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=candidates.csv"},
    )


@bp.route("/export/audits")
@hr_required
def export_audits():
    if not current_user.is_hr_admin:
        flash("审计导出仅限管理员", "error")
        return redirect(url_for("hr.dashboard"))
    fmt = (request.args.get("format") or "csv").lower()
    rows = (
        AuditLog.query.filter(
            (AuditLog.org_id == current_user.org_id) | (AuditLog.org_id.is_(None))
        )
        .order_by(AuditLog.id.desc())
        .limit(2000)
        .all()
    )
    audit("export_audits", "export", detail={"format": fmt, "count": len(rows)})
    if fmt == "pdf":
        data = audits_to_pdf_bytes(rows)
        return Response(
            data,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=audits.pdf"},
        )
    return Response(
        "\ufeff" + audits_to_csv(rows),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=audits.csv"},
    )


@bp.route("/jobs", methods=["GET", "POST"])
@hr_required
def jobs():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        department = (request.form.get("department") or "").strip()
        requirements = (request.form.get("requirements") or "").strip()
        if not title or not requirements:
            flash("岗位名称与需求必填", "error")
        else:
            job = JobPosting(
                org_id=current_user.org_id,
                title=title,
                department=department,
                requirements=requirements,
                is_active=True,
            )
            db.session.add(job)
            db.session.commit()
            audit("job_created", "job_posting", job.id, detail={"title": title})
            flash("岗位已发布，候选人上传时可选择", "success")
            return redirect(url_for("hr.jobs"))

    items = (
        JobPosting.query.filter_by(org_id=current_user.org_id)
        .order_by(JobPosting.created_at.desc())
        .all()
    )
    return render_template("hr/jobs.html", jobs=items)


@bp.route("/jobs/<int:job_id>/toggle", methods=["POST"])
@hr_required
def job_toggle(job_id: int):
    job = JobPosting.query.get_or_404(job_id)
    if job.org_id != current_user.org_id:
        abort(403)
    job.is_active = not job.is_active
    db.session.commit()
    audit("job_toggled", "job_posting", job.id, detail={"is_active": job.is_active})
    flash(("已启用" if job.is_active else "已停用") + f"岗位「{job.title}」", "success")
    return redirect(url_for("hr.jobs"))


@bp.route("/settings/notify", methods=["GET", "POST"])
@hr_required
def settings_notify():
    if request.method == "POST":
        current_user.email = (request.form.get("email") or "").strip()
        current_user.phone = (request.form.get("phone") or "").strip()
        current_user.notify_email = bool(request.form.get("notify_email"))
        current_user.notify_sms = bool(request.form.get("notify_sms"))
        db.session.commit()
        if (request.form.get("action") or "") == "test":
            notify_user(
                current_user.id,
                title="测试通知",
                body="这是一条 HirePilot HR 测试消息。",
                link="/hr/settings/notify",
            )
            flash("已发送测试通知（请查站内通知与 data/outbox）", "success")
        else:
            flash("通知设置已保存", "success")
        return redirect(url_for("hr.settings_notify"))
    return render_template(
        "profile_notify.html",
        user=current_user,
        brand_href=url_for("hr.dashboard"),
        role="hr",
    )


@bp.route("/settings/weights", methods=["GET", "POST"])
@hr_required
def settings_weights():
    org = get_org(current_user.org_id)
    if not org:
        flash("未找到所属组织", "error")
        return redirect(url_for("hr.dashboard"))

    if request.method == "POST":
        try:
            saved = save_weights(
                org,
                rank_resume=parse_percent(request.form.get("rank_resume"), 70),
                rank_interview=parse_percent(request.form.get("rank_interview"), 30),
                dim={
                    "skill": parse_percent(request.form.get("weight_skill"), 25),
                    "project": parse_percent(request.form.get("weight_project"), 25),
                    "education": parse_percent(request.form.get("weight_education"), 25),
                    "fit": parse_percent(request.form.get("weight_fit"), 25),
                },
            )
            n = recompute_org_totals(org.id)
            db.session.commit()
            audit(
                "weights_updated",
                "organization",
                org.id,
                detail={**saved, "recomputed": n},
            )
            flash(
                f"权重已保存并归一化；已按新四维权重重算 {n} 份简历总分与达标状态。",
                "success",
            )
        except Exception:
            logger.exception("save weights failed")
            db.session.rollback()
            flash("保存权重失败", "error")
        return redirect(url_for("hr.settings_weights"))

    resume_w, interview_w = get_rank_weights(org.id)
    dims = get_dim_weights(org.id)
    return render_template(
        "hr/settings_weights.html",
        org=org,
        resume_w=resume_w,
        interview_w=interview_w,
        dims=dims,
        threshold=Config.RESUME_SCORE_THRESHOLD,
    )


@bp.route("/rag-eval")
@hr_required
def rag_eval_legacy_redirect():
    """Old HR URL → ops console (admin only)."""
    if current_user.is_hr_admin:
        return redirect(url_for("ops.rag_eval", **request.args))
    abort(404)


@bp.route("/rag-eval/run", methods=["POST"])
@hr_required
def rag_eval_run_legacy_redirect():
    if current_user.is_hr_admin:
        return redirect(url_for("ops.rag_eval_run"), code=307)
    abort(404)


@bp.route("/logs")
@hr_required
def backend_logs_legacy_redirect():
    if current_user.is_hr_admin:
        return redirect(url_for("ops.backend_logs"))
    abort(404)


@bp.route("/notifications")
@hr_required
def notifications():
    from app.services.notify import list_notifications

    items = list_notifications(current_user.id)
    return render_template(
        "notifications.html",
        notifications=items,
        mark_all_url=url_for("hr.notifications_read"),
    )


@bp.route("/notifications/read", methods=["POST"])
@hr_required
def notifications_read():
    from app.services.notify import mark_read

    nid = request.form.get("id", type=int)
    mark_read(current_user.id, nid)
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify({"ok": True})
    return redirect(url_for("hr.notifications"))


@bp.route("/chat", methods=["GET", "POST"])
@hr_required
def chat():
    sessions = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    sid = request.args.get("session_id", type=int)
    chat_session = None
    if sid:
        chat_session = ChatSession.query.get_or_404(sid)
        if chat_session.user_id != current_user.id:
            abort(403)

    if request.method == "POST":
        return redirect(url_for("hr.chat", session_id=sid))

    messages = chat_session.messages.all() if chat_session else []
    return render_template(
        "hr/chat.html",
        sessions=sessions,
        chat_session=chat_session,
        messages=messages,
        stream_url=url_for("hr.chat_stream"),
        delete_url_template=url_for("hr.chat_delete", session_id=0).rsplit("0", 1)[0],
    )


@bp.route("/chat/stream", methods=["POST"])
@hr_required
@limiter.limit(Config.RATELIMIT_CHAT)
def chat_stream():
    from app.services.chat_stream import stream_chat_response

    incr("chat_requests")
    data = request.get_json(silent=True) or {}
    content = (data.get("message") or request.form.get("message") or "").strip()
    session_id = data.get("session_id") or request.form.get("session_id", type=int)
    return stream_chat_response(
        user_id=current_user.id,
        role="hr",
        content=content,
        session_id=int(session_id) if session_id else None,
        context_note=hr_context_note(org_id=current_user.org_id),
        org_id=current_user.org_id,
    )


@bp.route("/chat/<int:session_id>/delete", methods=["POST"])
@hr_required
def chat_delete(session_id: int):
    from app.services.chat_stream import delete_chat_session

    ok = delete_chat_session(user_id=current_user.id, session_id=session_id)
    if not ok:
        abort(404)
    if request.headers.get("X-Requested-With") == "fetch" or request.is_json:
        return {"ok": True}
    flash("对话已删除", "success")
    return redirect(url_for("hr.chat"))

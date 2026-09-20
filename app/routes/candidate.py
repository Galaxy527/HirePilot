"""Candidate-only routes."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from app.agents.graph_runtime import invoke_agent, new_thread_id
from app.config import Config
from app.extensions import limiter
from app.models import ChatSession, InterviewSession, JobPosting, LiveInterview, Resume, db
from app.services import jobs as jobq
from app.services.audit import audit
from app.services.chat_context import candidate_context_note
from app.services.metrics import incr
from app.services.resume_parser import (
    UploadValidationError,
    extract_text_from_bytes,
    validate_upload,
)
from app.services.resume_service import create_resume_draft
from app.services.scoring import build_details_for_resume

logger = logging.getLogger(__name__)
bp = Blueprint("candidate", __name__, url_prefix="/candidate")


def candidate_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_candidate:
            abort(403)
        return fn(*args, **kwargs)

    return wrapper


@bp.route("/")
@candidate_required
def home():
    resumes = (
        Resume.query.filter_by(candidate_id=current_user.id)
        .order_by(Resume.created_at.desc())
        .all()
    )
    return render_template(
        "candidate/home.html",
        resumes=resumes,
        threshold=Config.RESUME_SCORE_THRESHOLD,
    )


@bp.route("/resume/<int:resume_id>/delete", methods=["POST"])
@candidate_required
def delete_resume(resume_id: int):
    from app.services.pii import delete_one_resume

    ok = delete_one_resume(
        resume_id, actor_id=current_user.id, candidate_id=current_user.id
    )
    if not ok:
        flash("简历不存在或无权删除", "error")
    else:
        flash(f"已删除简历 #{resume_id}", "success")
        incr("resume_deleted")
    return redirect(url_for("candidate.home"))


@bp.route("/resume/upload", methods=["GET", "POST"])
@candidate_required
@limiter.limit(Config.RATELIMIT_UPLOAD)
def upload_resume():
    from app.services.job_match import resolve_job_for_candidate

    jobs = (
        JobPosting.query.filter_by(is_active=True, org_id=current_user.org_id)
        .order_by(JobPosting.id.asc())
        .all()
    )
    if request.method == "GET":
        return render_template("candidate/upload.html", jobs=jobs)

    paste = (request.form.get("paste_text") or "").strip()
    file = request.files.get("file")
    job_id = request.form.get("job_id", type=int)

    selected_job, job_title, job_requirements, match_note = resolve_job_for_candidate(
        job_id=job_id, org_id=current_user.org_id
    )
    if not selected_job or not job_requirements:
        flash(match_note or "请选择 HR 已发布的求职岗位", "error")
        return render_template("candidate/upload.html", jobs=jobs, form=request.form), 400

    raw_text = ""
    filename = None

    if file and file.filename:
        try:
            filename, data = validate_upload(file, allowed_ext=Config.ALLOWED_EXTENSIONS)
            dest = Config.UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{filename}"
            dest.write_bytes(data)
            raw_text = extract_text_from_bytes(filename, data)
        except UploadValidationError as exc:
            flash(str(exc), "error")
            return render_template("candidate/upload.html", jobs=jobs), 400
        except Exception as exc:
            logger.exception("Parse failed: %s", exc)
            flash(f"解析失败：{exc}", "error")
            return render_template("candidate/upload.html", jobs=jobs), 400
    elif paste:
        raw_text = paste
        filename = "pasted.txt"
    else:
        flash("请上传文件或粘贴文本", "error")
        return render_template("candidate/upload.html", jobs=jobs), 400

    if not raw_text.strip():
        flash("未能提取到文本（扫描版 PDF 需 OCR，本期未启用）", "error")
        return render_template("candidate/upload.html", jobs=jobs), 400

    resume = create_resume_draft(
        candidate_id=current_user.id,
        raw_text=raw_text,
        filename=filename,
        job_id=selected_job.id,
        job_title=job_title,
        job_requirements=job_requirements,
        org_id=current_user.org_id,
    )
    jobq.enqueue(
        jobq.KIND_SCORE_RESUME,
        {"resume_id": resume.id, "candidate_name": current_user.display_name},
    )
    incr("resumes_uploaded")
    audit(
        "resume_uploaded",
        "resume",
        resume.id,
        detail={"job_id": selected_job.id, "job_title": job_title},
        org_id=current_user.org_id,
    )
    flash(f"{match_note}，已提交评分队列", "success")
    return redirect(url_for("candidate.resume_detail", resume_id=resume.id))


@bp.route("/resume/<int:resume_id>")
@candidate_required
def resume_detail(resume_id: int):
    resume = Resume.query.get_or_404(resume_id)
    if resume.candidate_id != current_user.id:
        abort(403)
    interviews = resume.interviews.order_by(InterviewSession.round_no).all()
    score_report = build_details_for_resume(resume) if not resume.is_processing else {}
    if not resume.is_processing and not resume.score_details() and score_report:
        resume.set_score_details(score_report)
        db.session.commit()
    return render_template(
        "candidate/resume_detail.html",
        resume=resume,
        interviews=interviews,
        threshold=Config.RESUME_SCORE_THRESHOLD,
        max_interviews=Config.MAX_INTERVIEWS_PER_RESUME,
        score_report=score_report,
    )


@bp.route("/resume/<int:resume_id>/status")
@candidate_required
def resume_status(resume_id: int):
    resume = Resume.query.get_or_404(resume_id)
    if resume.candidate_id != current_user.id:
        abort(403)
    return jsonify(
        {
            "id": resume.id,
            "processing_status": resume.processing_status,
            "total_score": resume.total_score,
            "meets_threshold": resume.meets_threshold,
            "ready": resume.processing_status == "ready",
            "failed": resume.processing_status == "failed",
        }
    )


@bp.route("/settings/notify", methods=["GET", "POST"])
@candidate_required
def notify_settings():
    if request.method == "POST":
        current_user.email = (request.form.get("email") or "").strip()
        current_user.phone = (request.form.get("phone") or "").strip()
        current_user.notify_email = bool(request.form.get("notify_email"))
        current_user.notify_sms = bool(request.form.get("notify_sms"))
        db.session.commit()
        if (request.form.get("action") or "") == "test":
            from app.services.notify import notify_user

            notify_user(
                current_user.id,
                title="测试通知",
                body="这是一条 HirePilot 测试消息（站内 + 邮件/短信通道）。",
                link="/candidate/settings/notify",
            )
            flash("已发送测试通知（请查站内通知与 data/outbox）", "success")
        else:
            flash("通知设置已保存", "success")
        return redirect(url_for("candidate.notify_settings"))
    return render_template(
        "profile_notify.html",
        user=current_user,
        brand_href=url_for("candidate.home"),
        role="candidate",
    )


@bp.route("/privacy/delete", methods=["GET", "POST"])
@candidate_required
def privacy_delete():
    if request.method == "GET":
        return render_template(
            "candidate/privacy_delete.html",
            retention_days=Config.PII_RETENTION_DAYS,
        )
    confirm = (request.form.get("confirm") or "").strip()
    if confirm != "DELETE":
        flash("请输入 DELETE 确认删除", "error")
        return render_template(
            "candidate/privacy_delete.html",
            retention_days=Config.PII_RETENTION_DAYS,
        ), 400
    from app.services.pii import delete_candidate_data

    stats = delete_candidate_data(current_user.id, actor_id=current_user.id)
    flash(f"已删除个人数据：简历 {stats['resumes']} 份", "success")
    return redirect(url_for("candidate.home"))


@bp.route("/history")
@candidate_required
def history():
    resumes = (
        Resume.query.filter_by(candidate_id=current_user.id)
        .order_by(Resume.created_at.desc())
        .all()
    )
    return render_template("candidate/history.html", resumes=resumes)


@bp.route("/interview")
@candidate_required
def interview_hub():
    resumes = (
        Resume.query.filter_by(candidate_id=current_user.id)
        .order_by(Resume.created_at.desc())
        .all()
    )
    max_n = Config.MAX_INTERVIEWS_PER_RESUME
    eligible = [
        r
        for r in resumes
        if r.meets_threshold and r.remaining_interviews(max_n) > 0 and not r.is_processing
    ]
    resume_ids = [r.id for r in resumes]
    active_sessions = []
    completed = []
    if resume_ids:
        active_sessions = (
            InterviewSession.query.filter(
                InterviewSession.resume_id.in_(resume_ids),
                InterviewSession.status == "in_progress",
            )
            .order_by(InterviewSession.created_at.desc())
            .all()
        )
        completed = (
            InterviewSession.query.filter(
                InterviewSession.resume_id.in_(resume_ids),
                InterviewSession.status == "completed",
            )
            .order_by(InterviewSession.completed_at.desc())
            .all()
        )
    return render_template(
        "candidate/interview_hub.html",
        resumes=resumes,
        eligible=eligible,
        active_sessions=active_sessions,
        completed=completed,
        threshold=Config.RESUME_SCORE_THRESHOLD,
        max_interviews=max_n,
    )


@bp.route("/interview/<int:resume_id>/start", methods=["POST"])
@candidate_required
def interview_start(resume_id: int):
    resume = Resume.query.get_or_404(resume_id)
    if resume.candidate_id != current_user.id:
        abort(403)
    if resume.is_processing:
        flash("简历仍在评分中，请稍后再试", "error")
        return redirect(url_for("candidate.resume_detail", resume_id=resume.id))
    if not resume.meets_threshold:
        flash(f"简历分未达阈值 {Config.RESUME_SCORE_THRESHOLD}，不可开启面试", "error")
        return redirect(url_for("candidate.resume_detail", resume_id=resume.id))
    if resume.remaining_interviews(Config.MAX_INTERVIEWS_PER_RESUME) <= 0:
        flash("本简历面试次数已用尽（最多 3 次）", "error")
        return redirect(url_for("candidate.resume_detail", resume_id=resume.id))

    active = (
        InterviewSession.query.filter_by(resume_id=resume.id, status="in_progress")
        .order_by(InterviewSession.round_no.desc())
        .first()
    )
    if active:
        return redirect(url_for("candidate.interview_room", session_id=active.id))

    round_no = resume.interview_count() + 1
    thread_id = new_thread_id()
    session = InterviewSession(
        resume_id=resume.id,
        round_no=round_no,
        status="in_progress",
        current_index=0,
        checkpoint_thread_id=thread_id,
    )
    session.set_questions([])
    session.set_qa_list([])
    db.session.add(session)
    db.session.commit()

    jobq.enqueue(jobq.KIND_GENERATE_INTERVIEW, {"session_id": session.id})
    audit("interview_started", "interview_session", session.id, detail={"resume_id": resume.id})
    flash("面试已创建，AI 正在根据简历出题…", "success")
    return redirect(url_for("candidate.interview_room", session_id=session.id))


@bp.route("/interview/<int:session_id>", methods=["GET", "POST"])
@candidate_required
def interview_room(session_id: int):
    session = InterviewSession.query.get_or_404(session_id)
    resume = session.resume
    if resume.candidate_id != current_user.id:
        abort(403)

    questions = session.questions()
    preparing = session.status == "in_progress" and not questions

    if request.method == "POST" and session.status == "in_progress" and questions:
        answer = (request.form.get("answer") or "").strip()
        if not answer:
            flash("请填写回答", "error")
        else:
            result = invoke_agent(
                {
                    "intent": "interview_evaluate",
                    "role": "candidate",
                    "user_id": current_user.id,
                    "questions": session.questions(),
                    "qa_list": session.qa_list(),
                    "current_index": session.current_index,
                    "user_answer": answer,
                    "resume_summary": resume.raw_text[:2000],
                    "structured": resume.structured(),
                    "query": answer,
                },
                thread_id=session.checkpoint_thread_id or new_thread_id(),
            )
            session.set_qa_list(result.get("qa_list") or [])
            session.current_index = int(result.get("current_index") or 0)
            if result.get("done"):
                session.status = "completed"
                session.score = result.get("score")
                session.report = result.get("report")
                session.completed_at = datetime.now(timezone.utc)
                try:
                    from app.services.rag import get_rag
                    from app.services.notify import notify_hrs

                    get_rag().index_resume(
                        resume_id=resume.id,
                        candidate_id=resume.candidate_id,
                        raw_text=resume.raw_text or "",
                        structured=resume.structured(),
                        candidate_name=current_user.display_name,
                        org_id=resume.org_id,
                        job_title=resume.target_job_title,
                        extra_note=(
                            f"面试第{session.round_no}场报告：\n{session.report or ''}"
                        ),
                    )
                    notify_hrs(
                        title="候选人完成模拟面试",
                        body=f"{current_user.display_name} · 简历 #{resume.id} · 场次 {session.round_no} · 分 {session.score}",
                        link=f"/hr/resume/{resume.id}",
                        org_id=resume.org_id,
                    )
                    if resume.pipeline_stage in ("new", "screening"):
                        resume.pipeline_stage = "interview"
                except Exception:
                    logger.exception("Failed to reindex after interview")
            db.session.commit()
            if session.status == "completed":
                flash("本场面试已完成", "success")
                return redirect(url_for("candidate.interview_report", session_id=session.id))

    questions = session.questions()
    idx = session.current_index
    current_q = questions[idx] if idx < len(questions) and session.status == "in_progress" else None
    return render_template(
        "candidate/interview.html",
        session=session,
        resume=resume,
        current_question=current_q,
        total=len(questions),
        preparing=preparing,
    )


@bp.route("/interview/<int:session_id>/status")
@candidate_required
def interview_status(session_id: int):
    session = InterviewSession.query.get_or_404(session_id)
    if session.resume.candidate_id != current_user.id:
        abort(403)
    qs = session.questions()
    return jsonify(
        {
            "id": session.id,
            "ready": bool(qs),
            "question_count": len(qs),
            "status": session.status,
        }
    )


@bp.route("/interview/<int:session_id>/report")
@candidate_required
def interview_report(session_id: int):
    session = InterviewSession.query.get_or_404(session_id)
    if session.resume.candidate_id != current_user.id:
        abort(403)
    return render_template("candidate/interview_report.html", session=session)


@bp.route("/notifications")
@candidate_required
def notifications():
    from app.services.notify import list_notifications, mark_read

    items = list_notifications(current_user.id)
    return render_template("notifications.html", notifications=items, mark_all_url=url_for("candidate.notifications_read"))


@bp.route("/notifications/read", methods=["POST"])
@candidate_required
def notifications_read():
    from app.services.notify import mark_read

    nid = request.form.get("id", type=int)
    mark_read(current_user.id, nid)
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify({"ok": True})
    return redirect(url_for("candidate.notifications"))


@bp.route("/chat", methods=["GET", "POST"])
@candidate_required
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
        return redirect(url_for("candidate.chat", session_id=sid))

    messages = chat_session.messages.all() if chat_session else []
    return render_template(
        "candidate/chat.html",
        sessions=sessions,
        chat_session=chat_session,
        messages=messages,
        stream_url=url_for("candidate.chat_stream"),
        delete_url_template=url_for("candidate.chat_delete", session_id=0).rsplit("0", 1)[0],
    )


@bp.route("/calendar")
@candidate_required
def calendar():
    resume_ids = [r.id for r in Resume.query.filter_by(candidate_id=current_user.id).all()]
    items = []
    if resume_ids:
        items = (
            LiveInterview.query.filter(LiveInterview.resume_id.in_(resume_ids))
            .order_by(LiveInterview.starts_at.asc())
            .all()
        )
    return render_template("candidate/calendar.html", items=items)


@bp.route("/chat/stream", methods=["POST"])
@candidate_required
@limiter.limit(Config.RATELIMIT_CHAT)
def chat_stream():
    from app.services.chat_stream import stream_chat_response

    incr("chat_requests")
    data = request.get_json(silent=True) or {}
    content = (data.get("message") or request.form.get("message") or "").strip()
    session_id = data.get("session_id") or request.form.get("session_id", type=int)
    return stream_chat_response(
        user_id=current_user.id,
        role="candidate",
        content=content,
        session_id=int(session_id) if session_id else None,
        context_note=candidate_context_note(current_user.id),
        candidate_id=current_user.id,
        org_id=current_user.org_id,
    )


@bp.route("/chat/<int:session_id>/delete", methods=["POST"])
@candidate_required
def chat_delete(session_id: int):
    from app.services.chat_stream import delete_chat_session

    ok = delete_chat_session(user_id=current_user.id, session_id=session_id)
    if not ok:
        abort(404)
    if request.headers.get("X-Requested-With") == "fetch" or request.is_json:
        return {"ok": True}
    flash("对话已删除", "success")
    return redirect(url_for("candidate.chat"))

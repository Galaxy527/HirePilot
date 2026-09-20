"""SQLAlchemy models for HirePilot."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Organization(db.Model):
    __tablename__ = "organizations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    code = db.Column(db.String(40), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=utcnow)

    # HR-configurable ranking: composite = resume_w * resume + interview_w * best_interview
    rank_resume_weight = db.Column(db.Float, default=0.7)
    rank_interview_weight = db.Column(db.Float, default=0.3)
    # HR-configurable resume dimension weights (must sum ~1)
    weight_skill = db.Column(db.Float, default=0.25)
    weight_project = db.Column(db.Float, default=0.25)
    weight_education = db.Column(db.Float, default=0.25)
    weight_fit = db.Column(db.Float, default=0.25)

    users = db.relationship("User", back_populates="organization", lazy="dynamic")
    jobs = db.relationship("JobPosting", back_populates="organization", lazy="dynamic")


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # candidate | hr
    hr_level = db.Column(db.String(20))  # admin | recruiter (HR only)
    org_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    email = db.Column(db.String(200), default="")
    phone = db.Column(db.String(40), default="")
    notify_email = db.Column(db.Boolean, default=True)
    notify_sms = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    organization = db.relationship("Organization", back_populates="users")
    resumes = db.relationship(
        "Resume",
        back_populates="candidate",
        lazy="dynamic",
        foreign_keys="Resume.candidate_id",
    )
    chat_sessions = db.relationship("ChatSession", back_populates="user", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_candidate(self) -> bool:
        return self.role == "candidate"

    @property
    def is_hr(self) -> bool:
        return self.role == "hr"

    @property
    def is_hr_admin(self) -> bool:
        return self.is_hr and (self.hr_level or "recruiter") == "admin"

    @property
    def is_hr_recruiter(self) -> bool:
        return self.is_hr and (self.hr_level or "recruiter") == "recruiter"


class JobPosting(db.Model):
    """企业发布的招聘岗位（评分与面试的参照标准）。"""

    __tablename__ = "job_postings"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    title = db.Column(db.String(200), nullable=False)
    department = db.Column(db.String(120), default="")
    requirements = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, default=True, index=True)
    created_at = db.Column(db.DateTime, default=utcnow)

    organization = db.relationship("Organization", back_populates="jobs")
    resumes = db.relationship("Resume", back_populates="job", lazy="dynamic")

    def profile_text(self) -> str:
        dept = f"（{self.department}）" if self.department else ""
        return f"目标岗位：{self.title}{dept}\n岗位需求：\n{self.requirements}"


class Resume(db.Model):
    __tablename__ = "resumes"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_postings.id"), nullable=True, index=True)
    target_job_title = db.Column(db.String(200))
    target_job_requirements = db.Column(db.Text)
    filename = db.Column(db.String(255))
    raw_text = db.Column(db.Text, nullable=False)
    structured_json = db.Column(db.Text)  # JSON string

    # Scores
    total_score = db.Column(db.Float)
    skill_score = db.Column(db.Float)
    project_score = db.Column(db.Float)
    education_score = db.Column(db.Float)
    fit_score = db.Column(db.Float)
    recommendation = db.Column(db.String(20))  # 推荐 / 待定 / 不推荐
    recommendation_reason = db.Column(db.Text)
    score_detail_json = db.Column(db.Text)
    meets_threshold = db.Column(db.Boolean, default=False, index=True)
    processing_status = db.Column(db.String(20), default="ready", index=True)
    pipeline_stage = db.Column(db.String(20), default="new", index=True)
    reject_reason = db.Column(db.String(40))
    reject_note = db.Column(db.Text)
    assigned_hr_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    candidate = db.relationship("User", back_populates="resumes", foreign_keys=[candidate_id])
    assigned_hr = db.relationship("User", foreign_keys=[assigned_hr_id])
    job = db.relationship("JobPosting", back_populates="resumes")
    interviews = db.relationship(
        "InterviewSession", back_populates="resume", lazy="dynamic", order_by="InterviewSession.round_no"
    )
    notes = db.relationship(
        "ResumeNote", back_populates="resume", lazy="dynamic", order_by="ResumeNote.created_at.desc()"
    )
    live_interviews = db.relationship(
        "LiveInterview", back_populates="resume", lazy="dynamic", order_by="LiveInterview.starts_at"
    )

    @property
    def is_processing(self) -> bool:
        return (self.processing_status or "") in {"queued", "scoring"}

    def job_profile_text(self) -> str:
        title = self.target_job_title or (self.job.title if self.job else "") or "未指定岗位"
        req = self.target_job_requirements or (self.job.requirements if self.job else "") or ""
        return f"目标岗位：{title}\n岗位需求：\n{req}".strip()


    def structured(self) -> dict:
        if not self.structured_json:
            return {}
        try:
            return json.loads(self.structured_json)
        except json.JSONDecodeError:
            return {}

    def set_structured(self, data: dict) -> None:
        self.structured_json = json.dumps(data, ensure_ascii=False)

    def score_details(self) -> dict:
        if not self.score_detail_json:
            return {}
        try:
            return json.loads(self.score_detail_json)
        except json.JSONDecodeError:
            return {}

    def set_score_details(self, data: dict) -> None:
        self.score_detail_json = json.dumps(data, ensure_ascii=False) if data else None

    def interview_count(self) -> int:
        return self.interviews.count()

    def remaining_interviews(self, max_n: int = 3) -> int:
        return max(0, max_n - self.interview_count())

    def best_interview_score(self) -> float | None:
        scores = [s.score for s in self.interviews if s.score is not None and s.status == "completed"]
        return max(scores) if scores else None

    def composite_score(self, resume_w: float = 0.7, interview_w: float = 0.3) -> float:
        """HR ranking: with interview use weighted mix; else resume only."""
        base = self.total_score or 0.0
        best = self.best_interview_score()
        if best is None:
            return base
        return resume_w * base + interview_w * best


class InterviewSession(db.Model):
    __tablename__ = "interview_sessions"

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False, index=True)
    round_no = db.Column(db.Integer, nullable=False)  # 1..3
    status = db.Column(db.String(20), default="in_progress")  # in_progress | completed
    questions_json = db.Column(db.Text)  # list of question strings
    qa_json = db.Column(db.Text)  # list of {q, a, eval}
    current_index = db.Column(db.Integer, default=0)
    score = db.Column(db.Float)
    report = db.Column(db.Text)
    checkpoint_thread_id = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=utcnow)
    completed_at = db.Column(db.DateTime)

    resume = db.relationship("Resume", back_populates="interviews")

    def questions(self) -> list:
        if not self.questions_json:
            return []
        return json.loads(self.questions_json)

    def set_questions(self, qs: list) -> None:
        self.questions_json = json.dumps(qs, ensure_ascii=False)

    def qa_list(self) -> list:
        if not self.qa_json:
            return []
        return json.loads(self.qa_json)

    def set_qa_list(self, items: list) -> None:
        self.qa_json = json.dumps(items, ensure_ascii=False)


class ChatSession(db.Model):
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(200), default="新对话")
    thread_id = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", back_populates="chat_sessions")
    messages = db.relationship(
        "Message", back_populates="session", lazy="dynamic", order_by="Message.created_at"
    )


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # user | assistant | system
    content = db.Column(db.Text, nullable=False)
    sources_json = db.Column(db.Text)  # optional RAG citations
    created_at = db.Column(db.DateTime, default=utcnow)

    session = db.relationship("ChatSession", back_populates="messages")

    def sources(self) -> list:
        if not self.sources_json:
            return []
        return json.loads(self.sources_json)

    def set_sources(self, srcs: list) -> None:
        self.sources_json = json.dumps(srcs, ensure_ascii=False) if srcs else None


class BackgroundJob(db.Model):
    __tablename__ = "background_jobs"

    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(64), nullable=False, index=True)
    payload_json = db.Column(db.Text, nullable=False, default="{}")
    status = db.Column(db.String(20), default="pending", index=True)
    # pending | running | succeeded | failed | dead
    progress = db.Column(db.Integer, default=0)
    error = db.Column(db.Text)
    attempts = db.Column(db.Integer, default=0)
    max_attempts = db.Column(db.Integer, default=3)
    next_run_at = db.Column(db.DateTime, default=utcnow, index=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    finished_at = db.Column(db.DateTime)

    def payload(self) -> dict:
        try:
            return json.loads(self.payload_json or "{}")
        except json.JSONDecodeError:
            return {}

    def set_payload(self, data: dict) -> None:
        self.payload_json = json.dumps(data or {}, ensure_ascii=False)


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    action = db.Column(db.String(80), nullable=False, index=True)
    entity_type = db.Column(db.String(40), nullable=False)
    entity_id = db.Column(db.String(64))
    detail_json = db.Column(db.Text)
    ip = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=utcnow)

    actor = db.relationship("User")


class ResumeNote(db.Model):
    __tablename__ = "resume_notes"

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False, index=True)
    author_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    resume = db.relationship("Resume", back_populates="notes")
    author = db.relationship("User")


class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, default="")
    link = db.Column(db.String(500), default="")
    is_read = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=utcnow)

    user = db.relationship("User")


class LiveInterview(db.Model):
    """Human interview scheduling (calendar), distinct from AI mock InterviewSession."""

    __tablename__ = "live_interviews"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    resume_id = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False, index=True)
    scheduled_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    interviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    title = db.Column(db.String(200), default="真人面试")
    starts_at = db.Column(db.DateTime, nullable=False, index=True)
    ends_at = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(255), default="")
    meeting_url = db.Column(db.String(500), default="")
    status = db.Column(db.String(20), default="scheduled", index=True)
    # scheduled | completed | cancelled
    notes = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=utcnow)

    resume = db.relationship("Resume", back_populates="live_interviews")
    scheduler = db.relationship("User", foreign_keys=[scheduled_by])
    interviewer = db.relationship("User", foreign_keys=[interviewer_id])

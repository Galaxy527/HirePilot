"""HirePilot Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask, jsonify
from flask_login import LoginManager

from app.config import Config
from app.extensions import csrf, limiter
from app.models import User, db
from app.routes import auth_bp, candidate_bp, hr_bp, ops_bp
from app.services.logging_setup import setup_logging

login_manager = LoginManager()
login_manager.login_view = "auth.login"


def create_app() -> Flask:
    Config.ensure_dirs()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    app.config["RATELIMIT_STORAGE_URI"] = Config.RATELIMIT_STORAGE_URI

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    setup_logging(app, Config.LOG_DIR)

    app.register_blueprint(auth_bp)
    app.register_blueprint(candidate_bp)
    app.register_blueprint(hr_bp)
    app.register_blueprint(ops_bp)

    @app.get("/healthz")
    def healthz():
        from sqlalchemy import text

        from app.services import jobs as jobq

        db_ok = True
        try:
            db.session.execute(text("SELECT 1"))
        except Exception:
            db_ok = False
        try:
            from app.services import embeddings as emb_mod

            if emb_mod._embedder is not None:
                emb = emb_mod._embedder
                embedder_info = {"name": emb.name, "dim": emb.dim, "loaded": True}
            else:
                embedder_info = {
                    "backend": Config.EMBEDDING_BACKEND,
                    "model": Config.LOCAL_EMBEDDING_MODEL
                    if Config.EMBEDDING_BACKEND == "local"
                    else Config.EMBEDDING_MODEL,
                    "loaded": False,
                }
        except Exception as exc:
            embedder_info = {"error": str(exc)}
        status = 200 if db_ok else 503
        return (
            jsonify(
                {
                    "status": "ok" if db_ok else "degraded",
                    "db": db_ok,
                    "llm": Config.llm_enabled(),
                    "inline_jobs": Config.INLINE_JOBS,
                    "jobs_pending": jobq.pending_count() if db_ok else None,
                    "jobs_dead": jobq.dead_letter_count() if db_ok else None,
                    "embedding": embedder_info,
                    "mail_log_only": Config.MAIL_LOG_ONLY,
                    "sms_log_only": Config.SMS_LOG_ONLY,
                }
            ),
            status,
        )

    @app.get("/metrics")
    def metrics():
        from app.services import jobs as jobq
        from app.services import metrics as metrics_svc

        data = metrics_svc.snapshot()
        data["jobs_pending"] = jobq.pending_count()
        data["jobs_dead"] = jobq.dead_letter_count()
        lines = [f"hirepilot_{k} {v}" for k, v in sorted(data.items())]
        return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; charset=utf-8"}

    csrf.exempt(healthz)
    csrf.exempt(metrics)

    @app.template_filter("strip_citations")
    def strip_citations_filter(text: str) -> str:
        import re

        if not text:
            return ""
        out = str(text)
        out = re.sub(r"\n*来源[：:].*$", "", out, flags=re.IGNORECASE | re.MULTILINE)
        out = re.sub(r"\n*引用来源[：:].*$", "", out, flags=re.IGNORECASE | re.MULTILINE)
        out = re.sub(r"\[简历#[^\]]+\]", "", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    with app.app_context():
        db.create_all()
        _ensure_schema()
        _seed_users()

    try:
        from app.agents.graph_runtime import reset_graph

        reset_graph()
    except Exception:
        pass

    @app.context_processor
    def inject_globals():
        from flask_login import current_user
        from flask_wtf.csrf import generate_csrf

        data = {"csrf_token": generate_csrf, "unread_notifications": 0}
        if current_user.is_authenticated:
            try:
                from app.services.notify import unread_count

                data["unread_notifications"] = unread_count(current_user.id)
            except Exception:
                pass
        return data

    logging.getLogger(__name__).info(
        "HirePilot ready. LLM=%s threshold=%s inline_jobs=%s",
        "on" if Config.llm_enabled() else "mock",
        Config.RESUME_SCORE_THRESHOLD,
        Config.INLINE_JOBS,
    )
    _warm_embeddings_async(app)
    return app


def _warm_embeddings_async(app: Flask) -> None:
    """Load embedder in background so the first chat request does not hang."""
    import threading

    def _run() -> None:
        with app.app_context():
            try:
                from app.services.embeddings import get_embedder

                emb = get_embedder()
                emb.embed_documents(["warmup"])
                logging.getLogger(__name__).info("Embedding warmup done: %s", emb.name)
            except Exception:
                logging.getLogger(__name__).exception("Embedding warmup failed")

    threading.Thread(target=_run, name="embed-warmup", daemon=True).start()


def _ensure_schema() -> None:
    """Add columns introduced after first create (SQLite)."""
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    log = logging.getLogger(__name__)

    def add_cols(table: str, wanted: dict[str, str]) -> None:
        if table not in tables:
            return
        cols = {c["name"] for c in inspector.get_columns(table)}
        alters = [stmt for name, stmt in wanted.items() if name not in cols]
        if not alters:
            return
        with db.engine.begin() as conn:
            for stmt in alters:
                conn.execute(text(stmt))
        log.info("Applied %s schema updates: %s", table, alters)

    add_cols(
        "users",
        {
            "org_id": "ALTER TABLE users ADD COLUMN org_id INTEGER",
            "hr_level": "ALTER TABLE users ADD COLUMN hr_level VARCHAR(20)",
            "email": "ALTER TABLE users ADD COLUMN email VARCHAR(200) DEFAULT ''",
            "phone": "ALTER TABLE users ADD COLUMN phone VARCHAR(40) DEFAULT ''",
            "notify_email": "ALTER TABLE users ADD COLUMN notify_email BOOLEAN DEFAULT 1",
            "notify_sms": "ALTER TABLE users ADD COLUMN notify_sms BOOLEAN DEFAULT 0",
        },
    )
    add_cols(
        "job_postings",
        {"org_id": "ALTER TABLE job_postings ADD COLUMN org_id INTEGER"},
    )
    add_cols(
        "resumes",
        {
            "score_detail_json": "ALTER TABLE resumes ADD COLUMN score_detail_json TEXT",
            "job_id": "ALTER TABLE resumes ADD COLUMN job_id INTEGER",
            "target_job_title": "ALTER TABLE resumes ADD COLUMN target_job_title VARCHAR(200)",
            "target_job_requirements": "ALTER TABLE resumes ADD COLUMN target_job_requirements TEXT",
            "processing_status": "ALTER TABLE resumes ADD COLUMN processing_status VARCHAR(20) DEFAULT 'ready'",
            "pipeline_stage": "ALTER TABLE resumes ADD COLUMN pipeline_stage VARCHAR(20) DEFAULT 'new'",
            "reject_reason": "ALTER TABLE resumes ADD COLUMN reject_reason VARCHAR(40)",
            "reject_note": "ALTER TABLE resumes ADD COLUMN reject_note TEXT",
            "assigned_hr_id": "ALTER TABLE resumes ADD COLUMN assigned_hr_id INTEGER",
            "org_id": "ALTER TABLE resumes ADD COLUMN org_id INTEGER",
        },
    )
    add_cols(
        "background_jobs",
        {
            "attempts": "ALTER TABLE background_jobs ADD COLUMN attempts INTEGER DEFAULT 0",
            "max_attempts": "ALTER TABLE background_jobs ADD COLUMN max_attempts INTEGER DEFAULT 3",
            "next_run_at": "ALTER TABLE background_jobs ADD COLUMN next_run_at DATETIME",
        },
    )
    add_cols(
        "audit_logs",
        {"org_id": "ALTER TABLE audit_logs ADD COLUMN org_id INTEGER"},
    )
    add_cols(
        "organizations",
        {
            "rank_resume_weight": "ALTER TABLE organizations ADD COLUMN rank_resume_weight FLOAT DEFAULT 0.7",
            "rank_interview_weight": "ALTER TABLE organizations ADD COLUMN rank_interview_weight FLOAT DEFAULT 0.3",
            "weight_skill": "ALTER TABLE organizations ADD COLUMN weight_skill FLOAT DEFAULT 0.25",
            "weight_project": "ALTER TABLE organizations ADD COLUMN weight_project FLOAT DEFAULT 0.25",
            "weight_education": "ALTER TABLE organizations ADD COLUMN weight_education FLOAT DEFAULT 0.25",
            "weight_fit": "ALTER TABLE organizations ADD COLUMN weight_fit FLOAT DEFAULT 0.25",
        },
    )


def _seed_users() -> None:
    from app.models import JobPosting, Organization
    from app.services.tenant import get_or_create_default_org

    org = get_or_create_default_org()

    if not User.query.filter_by(username="candidate1").first():
        c1 = User(
            username="candidate1",
            display_name="张三",
            role="candidate",
            org_id=org.id,
            email="candidate1@example.com",
            phone="13800000001",
            notify_email=True,
            notify_sms=True,
        )
        c1.set_password("demo123")
        c2 = User(
            username="candidate2",
            display_name="李四",
            role="candidate",
            org_id=org.id,
            email="candidate2@example.com",
            phone="13800000002",
            notify_email=True,
            notify_sms=False,
        )
        c2.set_password("demo123")
        hr = User(
            username="hr1",
            display_name="王HR",
            role="hr",
            hr_level="admin",
            org_id=org.id,
            email="hr1@example.com",
            phone="13900000001",
            notify_email=True,
            notify_sms=True,
        )
        hr.set_password("demo123")
        hr2 = User(
            username="hr2",
            display_name="赵招聘",
            role="hr",
            hr_level="recruiter",
            org_id=org.id,
            email="hr2@example.com",
            phone="13900000002",
            notify_email=True,
            notify_sms=False,
        )
        hr2.set_password("demo123")
        db.session.add_all([c1, c2, hr, hr2])
        db.session.commit()
        logging.getLogger(__name__).info("Seeded demo users candidate1/2 hr1(admin)/hr2(recruiter)")

    # Ensure recruiter demo account exists
    if not User.query.filter_by(username="hr2").first():
        hr2 = User(
            username="hr2",
            display_name="赵招聘",
            role="hr",
            hr_level="recruiter",
            org_id=org.id,
            email="hr2@example.com",
            phone="13900000002",
            notify_email=True,
        )
        hr2.set_password("demo123")
        db.session.add(hr2)
        db.session.commit()

    # Backfill contact info for demo users if empty
    demos = {
        "candidate1": ("candidate1@example.com", "13800000001"),
        "candidate2": ("candidate2@example.com", "13800000002"),
        "hr1": ("hr1@example.com", "13900000001"),
        "hr2": ("hr2@example.com", "13900000002"),
    }
    for uname, (em, ph) in demos.items():
        u = User.query.filter_by(username=uname).first()
        if u and not (u.email or "").strip():
            u.email = em
            u.phone = ph
            u.notify_email = True

    # Backfill org / hr_level on existing rows
    for u in User.query.filter(User.org_id.is_(None)).all():
        u.org_id = org.id
        if u.role == "hr" and not u.hr_level:
            u.hr_level = "admin" if u.username == "hr1" else "recruiter"
    hr1 = User.query.filter_by(username="hr1").first()
    if hr1 and not hr1.hr_level:
        hr1.hr_level = "admin"
    for j in JobPosting.query.filter(JobPosting.org_id.is_(None)).all():
        j.org_id = org.id
    from app.models import Resume

    for r in Resume.query.filter(Resume.org_id.is_(None)).all():
        r.org_id = org.id
    db.session.commit()

    if JobPosting.query.filter_by(org_id=org.id).count() == 0:
        db.session.add_all(
            [
                JobPosting(
                    org_id=org.id,
                    title="Python 后端工程师",
                    department="研发中心",
                    requirements=(
                        "职责：负责招聘/业务中台后端服务设计与开发。\n"
                        "必备：Python、Flask/FastAPI、SQL、Git、Linux。\n"
                        "加分：Docker、Redis、LangChain/Agent。"
                    ),
                ),
                JobPosting(
                    org_id=org.id,
                    title="前端开发工程师",
                    department="研发中心",
                    requirements=(
                        "职责：负责 Web 前端页面与交互实现。\n"
                        "必备：JavaScript/TypeScript、HTML/CSS、React 或 Vue。"
                    ),
                ),
                JobPosting(
                    org_id=org.id,
                    title="算法 / AI 应用工程师",
                    department="AI 组",
                    requirements=(
                        "职责：搭建 LLM 应用与 Agent 流程。\n"
                        "必备：Python、NLP 基础、RAG 实践。"
                    ),
                ),
            ]
        )
        db.session.commit()
        logging.getLogger(__name__).info("Seeded default job postings")

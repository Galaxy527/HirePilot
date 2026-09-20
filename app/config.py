"""Application configuration — secrets and tunables live in env / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "hirepilot-dev-secret")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'hirepilot.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session / CSRF
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "").strip() in {
        "1",
        "true",
        "True",
    }
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None

    RESUME_SCORE_THRESHOLD = float(os.getenv("RESUME_SCORE_THRESHOLD", "70"))

    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Embeddings: local (sentence-transformers) | api (OpenAI-compatible) | hash
    EMBEDDING_BACKEND = (os.getenv("EMBEDDING_BACKEND", "local") or "local").strip().lower()
    LOCAL_EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip() or "https://api.openai.com/v1"

    # Outbound notify
    MAIL_ENABLED = os.getenv("MAIL_ENABLED", "").strip() in {"1", "true", "True", "yes"}
    MAIL_SERVER = os.getenv("MAIL_SERVER", "localhost").strip()
    MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
    MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "1").strip() in {"1", "true", "True", "yes"}
    MAIL_USERNAME = os.getenv("MAIL_USERNAME", "").strip()
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "").strip()
    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", "noreply@hirepilot.local").strip()
    MAIL_LOG_ONLY = os.getenv("MAIL_LOG_ONLY", "1").strip() in {"1", "true", "True", "yes"}

    SMS_ENABLED = os.getenv("SMS_ENABLED", "").strip() in {"1", "true", "True", "yes"}
    SMS_PROVIDER = (os.getenv("SMS_PROVIDER", "log") or "log").strip().lower()  # log | twilio
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    SMS_LOG_ONLY = os.getenv("SMS_LOG_ONLY", "1").strip() in {"1", "true", "True", "yes"}
    APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "http://127.0.0.1:5000").rstrip("/")

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "data" / "uploads"))
    FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "data" / "faiss"))
    CHECKPOINT_DB = Path(
        os.getenv("CHECKPOINT_DB", BASE_DIR / "data" / "checkpoints" / "langgraph.db")
    )
    LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
    OUTBOX_DIR = Path(os.getenv("OUTBOX_DIR", BASE_DIR / "data" / "outbox"))

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
    MAX_INTERVIEWS_PER_RESUME = 3
    INTERVIEW_QUESTION_COUNT = 5

    RANK_RESUME_WEIGHT = 0.7
    RANK_INTERVIEW_WEIGHT = 0.3

    INLINE_JOBS = os.getenv("HIRING_INLINE_JOBS", "").strip() in {"1", "true", "True", "yes"}
    WORKER_POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "1.5"))
    JOB_MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", "3"))
    JOB_RETRY_BASE_SECONDS = float(os.getenv("JOB_RETRY_BASE_SECONDS", "5"))

    MIN_RETRIEVE_SCORE = float(os.getenv("MIN_RETRIEVE_SCORE", "0.08"))
    PII_RETENTION_DAYS = int(os.getenv("PII_RETENTION_DAYS", "180"))

    # Multi-tenant / HR invite
    DEFAULT_ORG_CODE = os.getenv("DEFAULT_ORG_CODE", "DEMO").strip().upper() or "DEMO"
    DEFAULT_ORG_NAME = os.getenv("DEFAULT_ORG_NAME", "HirePilot Demo Corp")
    HR_INVITE_CODE = os.getenv("HR_INVITE_CODE", "HR-DEMO-2026").strip()

    # Rate limits (Flask-Limiter strings)
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per hour")
    RATELIMIT_AUTH = os.getenv("RATELIMIT_AUTH", "20 per minute")
    RATELIMIT_UPLOAD = os.getenv("RATELIMIT_UPLOAD", "10 per minute")
    RATELIMIT_CHAT = os.getenv("RATELIMIT_CHAT", "30 per minute")
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    PIPELINE_STAGES = ("new", "screening", "interview", "offer", "hired", "rejected")
    REJECT_REASONS = (
        "skill_mismatch",
        "experience_short",
        "culture_fit",
        "salary",
        "duplicate",
        "other",
    )

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in (
            cls.UPLOAD_FOLDER,
            cls.FAISS_INDEX_DIR,
            cls.CHECKPOINT_DB.parent,
            cls.LOG_DIR,
            cls.OUTBOX_DIR,
            BASE_DIR / "data",
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def llm_enabled(cls) -> bool:
        return bool(cls.LLM_API_KEY)

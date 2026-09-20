"""Pytest fixtures for HirePilot."""
from __future__ import annotations

import os

import pytest

os.environ["LLM_API_KEY"] = ""
os.environ["HIRING_INLINE_JOBS"] = "1"
os.environ["FLASK_SECRET_KEY"] = "test-secret"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["MIN_RETRIEVE_SCORE"] = "0.08"
os.environ["HR_INVITE_CODE"] = "HR-DEMO-2026"
os.environ["DEFAULT_ORG_CODE"] = "DEMO"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["MAIL_LOG_ONLY"] = "1"
os.environ["SMS_LOG_ONLY"] = "1"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ckpt" / "lg.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("HIRING_INLINE_JOBS", "1")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("WTF_CSRF_ENABLED", "0")

    from app.config import Config

    Config.UPLOAD_FOLDER = tmp_path / "uploads"
    Config.FAISS_INDEX_DIR = tmp_path / "faiss"
    Config.CHECKPOINT_DB = tmp_path / "ckpt" / "lg.db"
    Config.LOG_DIR = tmp_path / "logs"
    Config.INLINE_JOBS = True
    Config.LLM_API_KEY = ""
    Config.WTF_CSRF_ENABLED = False
    Config.EMBEDDING_BACKEND = "hash"
    Config.MAIL_LOG_ONLY = True
    Config.SMS_LOG_ONLY = True
    Config.ensure_dirs()

    from app.services.embeddings import reset_embedder

    reset_embedder()

    from app import create_app

    application = create_app()
    application.config["TESTING"] = True
    application.config["WTF_CSRF_ENABLED"] = False

    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def candidate_client(client, app):
    client.post(
        "/login",
        data={"username": "candidate1", "password": "demo123", "role": "candidate"},
        follow_redirects=True,
    )
    return client


@pytest.fixture()
def hr_client(client, app):
    client.post(
        "/login",
        data={"username": "hr1", "password": "demo123", "role": "hr"},
        follow_redirects=True,
    )
    return client

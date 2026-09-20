"""Outbound notify + embedding smoke tests."""
from pathlib import Path


def test_email_sms_outbox(tmp_path, monkeypatch, app):
    with app.app_context():
        from app.config import Config
        from app.services.channels import send_email, send_sms

        Config.OUTBOX_DIR = tmp_path / "outbox"
        Config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        Config.MAIL_LOG_ONLY = True
        Config.SMS_LOG_ONLY = True
        assert send_email("a@b.com", subject="hi", body="hello")
        assert send_sms("13800000000", body="短信测试")
        files = list(Config.OUTBOX_DIR.glob("*.txt"))
        assert len(files) >= 2


def test_hash_embedder_dim():
    from app.services.embeddings import HashEmbedder

    emb = HashEmbedder(dim=64)
    vecs = emb.embed_documents(["Python Flask 简历", "hello world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 64


def test_notify_user_writes_outbox(tmp_path, monkeypatch, app):
    with app.app_context():
        from app.config import Config
        from app.models import User
        from app.services.notify import notify_user

        Config.OUTBOX_DIR = tmp_path / "outbox2"
        Config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        Config.MAIL_LOG_ONLY = True
        Config.SMS_LOG_ONLY = True
        u = User.query.filter_by(username="candidate1").first()
        u.email = "c1@test.com"
        u.phone = "13800138000"
        u.notify_email = True
        u.notify_sms = True
        from app.models import db

        db.session.commit()
        notify_user(u.id, title="单元测试", body="body", link="/x")
        assert list(Config.OUTBOX_DIR.glob("*.txt"))

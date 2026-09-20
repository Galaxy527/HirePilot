"""Auth / register validation tests."""


def test_register_rejects_short_username(client):
    rv = client.post(
        "/register",
        data={
            "username": "ab",
            "display_name": "测试用户",
            "password": "demo123",
            "confirm_password": "demo123",
            "role": "candidate",
        },
    )
    assert rv.status_code == 400


def test_register_and_login_candidate(client, app):
    rv = client.post(
        "/register",
        data={
            "username": "newcand01",
            "display_name": "新候选人",
            "password": "demo1234",
            "confirm_password": "demo1234",
            "role": "candidate",
            "org_code": "DEMO",
        },
        follow_redirects=False,
    )
    assert rv.status_code in (302, 303)
    with app.app_context():
        from app.models import User

        u = User.query.filter_by(username="newcand01").first()
        assert u is not None
        assert u.role == "candidate"
        assert u.org_id is not None


def test_register_hr_requires_invite(client):
    rv = client.post(
        "/register",
        data={
            "username": "hrnew01",
            "display_name": "新HR",
            "password": "demo1234",
            "confirm_password": "demo1234",
            "role": "hr",
            "hr_invite": "wrong",
            "org_code": "DEMO",
        },
    )
    assert rv.status_code == 400
    assert "邀请码" in rv.get_data(as_text=True)

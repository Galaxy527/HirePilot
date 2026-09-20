"""Auth routes — candidate / HR login & registration."""
from __future__ import annotations

import logging
import re

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user

from app.config import Config
from app.extensions import limiter
from app.models import User, db
from app.services.metrics import incr
from app.services.tenant import get_or_create_default_org, resolve_org_by_code

logger = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)

USERNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{2,19}$")
DISPLAY_NAME_MIN = 2
DISPLAY_NAME_MAX = 40
PASSWORD_MIN = 6
PASSWORD_MAX = 64


def _auth_page(mode: str = "login", preset_role: str = "candidate", form: dict | None = None, status: int = 200):
    return (
        render_template(
            "auth/login.html",
            mode=mode,
            preset_role=preset_role or "candidate",
            form=form or {},
            default_org_code=Config.DEFAULT_ORG_CODE,
        ),
        status,
    )


def _validate_register(
    username: str,
    display_name: str,
    password: str,
    confirm: str,
    role: str,
    *,
    hr_invite: str = "",
    org_code: str = "",
) -> tuple[str | None, object | None]:
    if role not in ("candidate", "hr"):
        return "请选择账号角色（候选人或 HR）", None
    if role == "hr":
        if (hr_invite or "").strip() != Config.HR_INVITE_CODE:
            return "HR 注册需要有效邀请码（公开注册仅支持候选人）", None
    if not USERNAME_RE.match(username):
        return "用户名须为 3–20 位，以字母开头，仅含字母、数字、下划线", None
    if not (DISPLAY_NAME_MIN <= len(display_name) <= DISPLAY_NAME_MAX):
        return f"显示名称须为 {DISPLAY_NAME_MIN}–{DISPLAY_NAME_MAX} 个字符", None
    if not (PASSWORD_MIN <= len(password) <= PASSWORD_MAX):
        return f"密码长度须为 {PASSWORD_MIN}–{PASSWORD_MAX} 位", None
    if password != confirm:
        return "两次输入的密码不一致", None
    if User.query.filter_by(username=username).first():
        return "该用户名已被注册", None
    org = resolve_org_by_code(org_code) if org_code else get_or_create_default_org()
    if org_code and not org:
        return "企业组织码无效", None
    if not org:
        org = get_or_create_default_org()
    return None, org


@bp.route("/")
def index():
    return redirect(url_for("auth.login"))


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit(Config.RATELIMIT_AUTH)
def login():
    if request.method == "GET":
        role = request.args.get("role", "candidate")
        mode = request.args.get("mode", "login")
        if mode not in ("login", "register"):
            mode = "login"
        return _auth_page(mode=mode, preset_role=role)[0]

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    role = (request.form.get("role") or "candidate").strip()

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        incr("auth_login_failed")
        flash("用户名或密码错误", "error")
        return _auth_page(mode="login", preset_role=role, form={"username": username}, status=401)

    if role and user.role != role:
        flash(f"该账号不是{('候选人' if role == 'candidate' else 'HR')}入口账号", "error")
        return _auth_page(mode="login", preset_role=role, form={"username": username}, status=403)

    login_user(user)
    incr("auth_login_ok")
    logger.info("User %s logged in as %s org=%s", user.id, user.role, user.org_id)
    if user.is_hr:
        return redirect(url_for("hr.dashboard"))
    return redirect(url_for("candidate.home"))


@bp.route("/register", methods=["GET", "POST"])
@limiter.limit(Config.RATELIMIT_AUTH)
def register():
    if request.method == "GET":
        role = request.args.get("role", "candidate")
        return _auth_page(mode="register", preset_role=role)[0]

    username = (request.form.get("username") or "").strip()
    display_name = (request.form.get("display_name") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""
    role = (request.form.get("role") or "candidate").strip()
    hr_invite = request.form.get("hr_invite") or ""
    org_code = (request.form.get("org_code") or "").strip()
    email = (request.form.get("email") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    form = {
        "username": username,
        "display_name": display_name,
        "org_code": org_code,
        "email": email,
        "phone": phone,
    }

    err, org = _validate_register(
        username,
        display_name,
        password,
        confirm,
        role,
        hr_invite=hr_invite,
        org_code=org_code,
    )
    if err:
        flash(err, "error")
        return _auth_page(mode="register", preset_role=role, form=form, status=400)

    user = User(
        username=username,
        display_name=display_name,
        role=role,
        org_id=org.id,
        hr_level="recruiter" if role == "hr" else None,
        email=email,
        phone=phone,
        notify_email=bool(email),
        notify_sms=bool(phone),
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    incr("auth_register_ok")
    logger.info("Registered user %s as %s org=%s", user.id, user.role, user.org_id)

    login_user(user)
    flash("注册成功，已自动登录", "success")
    if user.is_hr:
        return redirect(url_for("hr.dashboard"))
    return redirect(url_for("candidate.home"))


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

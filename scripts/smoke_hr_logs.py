"""Smoke: login as HR and hit /hr/ on the live server."""
from __future__ import annotations

import re
import sys

import requests

base = "http://127.0.0.1:5000"
s = requests.Session()
login_page = s.get(f"{base}/login", timeout=10)
token = ""
m = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login_page.text)
if not m:
    m = re.search(r'csrf-token" content="([^"]+)"', login_page.text)
if m:
    token = m.group(1)
r = s.post(
    f"{base}/login",
    data={"username": "hr1", "password": "demo123", "role": "hr", "csrf_token": token},
    timeout=10,
    allow_redirects=True,
)
print("after_login", r.status_code, r.url)
print("has_BuildError", "BuildError" in r.text)
print("has_logs_nav", "后端日志" in r.text or "/hr/logs" in r.text)
logs = s.get(f"{base}/hr/logs", timeout=10)
print("logs_page", logs.status_code, "BuildError" in logs.text)
sys.exit(0 if r.status_code == 200 and "BuildError" not in r.text else 1)

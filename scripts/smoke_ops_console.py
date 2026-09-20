"""Smoke: ops console is admin-only and not in HR nav."""
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
dash = s.get(f"{base}/hr/", timeout=10)
print("hr_nav_has_ops", ("RAG 评测" in dash.text) or ("后端日志" in dash.text) or ("/ops/" in dash.text and "运维" in dash.text))
print("hr_nav_clean", "后端日志" not in dash.text and "RAG 评测" not in dash.text)
ops = s.get(f"{base}/ops/", timeout=10)
logs = s.get(f"{base}/ops/logs", timeout=10)
rag = s.get(f"{base}/ops/rag-eval", timeout=10)
print("ops_home", ops.status_code, "BuildError" in ops.text)
print("ops_logs", logs.status_code, "BuildError" in logs.text)
print("ops_rag", rag.status_code, "BuildError" in rag.text)
ok = (
    r.status_code == 200
    and "BuildError" not in r.text
    and "后端日志" not in dash.text
    and "RAG 评测" not in dash.text
    and ops.status_code == 200
    and logs.status_code == 200
)
sys.exit(0 if ok else 1)

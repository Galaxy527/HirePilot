"""Run HirePilot: python run.py

Async scoring/interview:
  - Default HIRING_INLINE_JOBS=1 runs jobs inside the web process.
  - For production-like mode: set HIRING_INLINE_JOBS=0 and run
    `python -m app.worker` in a second terminal.

Hot reload (optional): set HIRING_RELOAD=1. Default off so long SSE
chat streams are not killed mid-request by watchdog restarts.
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    use_reloader = os.getenv("HIRING_RELOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=use_reloader,
        exclude_patterns=[
            "*/logs/*",
            "*/data/*",
            "*/.git/*",
            "*/tests/*",
            "*/__pycache__/*",
            "*/site-packages/*",
            "*.log",
            "*.db",
            "*.pyc",
        ],
    )

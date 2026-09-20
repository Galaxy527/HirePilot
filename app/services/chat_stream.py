"""Shared streaming chat helpers for candidate/HR routes."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from flask import Response, stream_with_context

from app.agents.graph_runtime import new_thread_id
from app.agents.qa_agent import stream_qa
from app.models import ChatSession, Message, db

logger = logging.getLogger(__name__)


def _strip_citations(text: str) -> str:
    import re

    out = text or ""
    out = re.sub(r"\n*来源[：:].*$", "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"\n*引用来源[：:].*$", "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"\[简历#[^\]]+\]", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chat_response(
    *,
    user_id: int,
    role: str,
    content: str,
    session_id: int | None,
    context_note: str,
    candidate_id: int | None = None,
    org_id: int | None = None,
) -> Response:
    content = (content or "").strip()
    if not content:
        return Response(
            sse_pack({"type": "error", "message": "请输入内容"}),
            mimetype="text/event-stream",
        )

    chat_session = None
    if session_id:
        chat_session = ChatSession.query.get(session_id)
        if not chat_session or chat_session.user_id != user_id:
            return Response(
                sse_pack({"type": "error", "message": "会话不存在"}),
                mimetype="text/event-stream",
            )
    if not chat_session:
        chat_session = ChatSession(
            user_id=user_id,
            title=content[:40],
            thread_id=new_thread_id(),
        )
        db.session.add(chat_session)
        db.session.flush()

    db.session.add(Message(session_id=chat_session.id, role="user", content=content))
    db.session.commit()
    sid = chat_session.id

    # Prior turns for resume-id / 指代 resolution
    prior = (
        Message.query.filter_by(session_id=sid)
        .order_by(Message.created_at.desc())
        .limit(8)
        .all()
    )
    recent_turns = "\n".join(
        f"{m.role}: {(m.content or '')[:240]}" for m in reversed(prior)
    )

    @stream_with_context
    def generate() -> Iterator[str]:
        # Emit meta immediately so the browser keeps the connection while RAG/LLM warms up.
        yield sse_pack({"type": "meta", "session_id": sid})
        yield sse_pack({"type": "status", "text": "正在检索与生成…"})
        parts: list[str] = []
        cite_sources: list = []
        try:
            token_iter, sources = stream_qa(
                {
                    "query": content,
                    "role": role,
                    "user_id": user_id,
                    "candidate_id": candidate_id or user_id,
                    "org_id": org_id,
                    "context_note": context_note,
                    "recent_turns": recent_turns,
                }
            )
            cite_sources = sources or []
            for piece in token_iter:
                parts.append(piece)
                yield sse_pack({"type": "token", "text": piece})
            full = "".join(parts).strip() or "暂时无法回答，请稍后再试。"
            # Don't strip citation markers from full dossier dumps
            if not full.startswith("======== 简历#"):
                full = _strip_citations(full)
            msg = Message(session_id=sid, role="assistant", content=full)
            if cite_sources and not full.startswith("不知道"):
                msg.set_sources(cite_sources)
            else:
                cite_sources = []
            db.session.add(msg)
            db.session.commit()
            yield sse_pack(
                {
                    "type": "done",
                    "session_id": sid,
                    "sources": cite_sources,
                }
            )
        except Exception as exc:
            logger.exception("stream chat error: %s", exc)
            db.session.rollback()
            yield sse_pack({"type": "error", "message": str(exc)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def delete_chat_session(*, user_id: int, session_id: int) -> bool:
    chat_session = ChatSession.query.get(session_id)
    if not chat_session or chat_session.user_id != user_id:
        return False
    Message.query.filter_by(session_id=session_id).delete()
    db.session.delete(chat_session)
    db.session.commit()
    return True

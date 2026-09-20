"""LLM helpers — real OpenAI-compatible client or deterministic mock."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from app.config import Config

logger = logging.getLogger(__name__)


def get_chat_llm(temperature: float | None = None):
    """Return a LangChain chat model when API key is set; else None."""
    if not Config.llm_enabled():
        logger.warning("LLM disabled: LLM_API_KEY empty")
        return None
    from langchain_openai import ChatOpenAI

    kwargs = {}
    model = (Config.LLM_MODEL or "").lower()
    if "deepseek" in model or "deepseek" in (Config.LLM_BASE_URL or "").lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        temperature=0.4 if temperature is None else temperature,
        timeout=90,
        **kwargs,
    )


def get_embeddings():
    """Backward-compatible: return LangChain-like object or None if hash-only."""
    from app.services.embeddings import get_embedder

    emb = get_embedder()
    if emb.name.startswith("hash"):
        return None
    return emb


def llm_text(prompt: str, fallback: str, *, temperature: float | None = None) -> str:
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        logger.warning("LLM text skipped (no client), using fallback")
        return fallback
    try:
        resp = llm.invoke(prompt)
        text = _message_text(resp)
        if not text:
            logger.warning("LLM text returned empty content, using fallback")
            return fallback
        return text
    except Exception as exc:
        logger.warning("LLM text call failed: %s", exc)
        return fallback


def llm_stream(prompt: str, *, temperature: float | None = None) -> Iterator[str]:
    """Yield text chunks from the chat model. Raises if LLM unavailable."""
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        raise RuntimeError("LLM not configured")
    for chunk in llm.stream(prompt):
        piece = _message_text(chunk)
        if piece:
            yield piece


def llm_json(prompt: str, fallback: dict | list, *, temperature: float | None = None) -> Any:
    """Ask LLM for JSON; fall back on parse failure or missing key."""
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return fallback
    try:
        resp = llm.invoke(prompt)
        text = _message_text(resp)
        if not text:
            logger.warning("LLM JSON returned empty content, using fallback")
            return fallback
        parsed = extract_json(text, None)
        if parsed is None:
            logger.warning("LLM JSON parse failed, raw=%s", text[:200])
            return fallback
        return parsed
    except Exception as exc:
        logger.warning("LLM JSON call failed: %s", exc)
        return fallback


def _message_text(resp) -> str:
    content = getattr(resp, "content", None)
    if content is None:
        return str(resp).strip() if resp is not None else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif hasattr(block, "get"):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts).strip()
    return str(content).strip()


def extract_json(text: str, fallback: Any) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"[\{\[][\s\S]*[\}\]]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return fallback

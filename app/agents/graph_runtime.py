"""主 Agent + LangGraph：意图路由、子 Agent 调度、Sqlite checkpoint。"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.interview_agent import (
    evaluate_answer,
    generate_questions,
    generate_report,
)
from app.agents.qa_agent import run_qa
from app.agents.resume_agent import run_resume_screening
from app.config import Config
from app.services.llm import llm_json

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: str
    role: str
    user_id: int
    candidate_id: int | None
    query: str
    raw_text: str
    resume_id: int
    org_id: int
    candidate_name: str
    job_title: str
    job_requirements: str
    structured: dict
    scores: dict
    answer: str
    sources: list
    resume_summary: str
    questions: list
    qa_list: list
    current_index: int
    action: str
    user_answer: str
    next_question: str
    score: float
    report: str
    done: bool
    error: str
    context_note: str
    questions_from_llm: bool
    final: str


def classify_intent(state: GraphState) -> GraphState:
    explicit = state.get("intent")
    if explicit and explicit != "auto":
        return {**state, "intent": explicit}

    query = state.get("query") or ""
    role = state.get("role") or "candidate"
    fallback = {"intent": "qa"}
    prompt = f"""判断用户意图，只输出 JSON：{{"intent":"qa"|"resume_screen"|"interview_generate"|"interview_evaluate"|"interview_report"|"chat"}}
角色={role}
用户输入：{query}
规则：问简历/分数/候选人数据 → qa；明确开始面试出题 → interview_generate；提交面试回答 → interview_evaluate；要报告 → interview_report；其它闲聊 → chat。
"""
    data = llm_json(prompt, fallback)
    intent = "qa"
    if isinstance(data, dict):
        intent = str(data.get("intent") or "qa")
    if intent not in {
        "qa",
        "resume_screen",
        "interview_generate",
        "interview_evaluate",
        "interview_report",
        "chat",
    }:
        intent = "qa"
    return {**state, "intent": intent}


def route_intent(
    state: GraphState,
) -> Literal[
    "resume_screen",
    "qa",
    "interview_generate",
    "interview_evaluate",
    "interview_report",
    "chat",
]:
    return state.get("intent") or "qa"  # type: ignore[return-value]


def node_resume(state: GraphState) -> GraphState:
    out = run_resume_screening(state)  # type: ignore[arg-type]
    final = "简历筛选完成。"
    if out.get("scores"):
        s = out["scores"]
        final = (
            f"简历评分完成：总分 {s.get('total_score')}，"
            f"推荐意见「{s.get('recommendation')}」。"
            f"{s.get('recommendation_reason') or ''}"
        )
    if out.get("error"):
        final = f"简历筛选失败：{out['error']}"
    return {**out, "final": final}


def node_qa(state: GraphState) -> GraphState:
    out = run_qa(state)  # type: ignore[arg-type]
    return {**out, "final": out.get("answer") or "不知道。"}


def node_interview_gen(state: GraphState) -> GraphState:
    out = generate_questions(state)  # type: ignore[arg-type]
    q = out.get("next_question") or ""
    return {**out, "final": f"面试开始。第 1 题：{q}"}


def node_interview_eval(state: GraphState) -> GraphState:
    out = evaluate_answer(state)  # type: ignore[arg-type]
    if out.get("done"):
        report_state = generate_report(out)  # type: ignore[arg-type]
        return {
            **report_state,
            "final": (
                f"本场面试已结束。综合分 {report_state.get('score')}。"
                f"\n{report_state.get('report')}"
            ),
        }
    idx = int(out.get("current_index") or 0)
    return {**out, "final": f"已记录。第 {idx + 1} 题：{out.get('next_question')}"}


def node_interview_report(state: GraphState) -> GraphState:
    out = generate_report(state)  # type: ignore[arg-type]
    return {**out, "final": out.get("report") or ""}


def node_chat(state: GraphState) -> GraphState:
    out = run_qa({**state, "query": state.get("query") or ""})  # type: ignore[arg-type]
    ans = out.get("answer") or "暂时无法回答，请稍后再试。"
    return {**out, "final": ans}


_checkpointer = None
_graph = None


def reset_graph() -> None:
    """Allow reloading graph after code changes in long-lived processes."""
    global _graph, _checkpointer
    _graph = None
    _checkpointer = None


def _build_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    Config.ensure_dirs()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(Config.CHECKPOINT_DB), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        _checkpointer.setup()
        logger.info("Using SqliteSaver checkpoint at %s", Config.CHECKPOINT_DB)
    except Exception as exc:
        logger.warning("Sqlite checkpoint unavailable (%s), using MemorySaver", exc)
        _checkpointer = MemorySaver()
    return _checkpointer


def build_graph():
    global _graph
    if _graph is not None:
        return _graph

    g = StateGraph(GraphState)
    g.add_node("classify", classify_intent)
    g.add_node("resume_screen", node_resume)
    g.add_node("qa", node_qa)
    g.add_node("interview_generate", node_interview_gen)
    g.add_node("interview_evaluate", node_interview_eval)
    g.add_node("interview_report", node_interview_report)
    g.add_node("chat", node_chat)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        route_intent,
        {
            "resume_screen": "resume_screen",
            "qa": "qa",
            "interview_generate": "interview_generate",
            "interview_evaluate": "interview_evaluate",
            "interview_report": "interview_report",
            "chat": "chat",
        },
    )
    for node in (
        "resume_screen",
        "qa",
        "interview_generate",
        "interview_evaluate",
        "interview_report",
        "chat",
    ):
        g.add_edge(node, END)

    _graph = g.compile(checkpointer=_build_checkpointer())
    logger.info("LangGraph compiled with checkpointing")
    return _graph


def new_thread_id() -> str:
    return uuid.uuid4().hex


def invoke_agent(payload: dict, thread_id: str | None = None) -> dict:
    graph = build_graph()
    tid = thread_id or new_thread_id()
    config = {"configurable": {"thread_id": tid}}
    logger.info("Agent invoke intent=%s thread=%s", payload.get("intent"), tid)
    result = graph.invoke(payload, config=config)
    result["_thread_id"] = tid
    return result

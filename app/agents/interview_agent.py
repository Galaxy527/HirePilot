"""模拟面试子 Agent — LLM 现场出题、评估、报告（无题库抽题）。"""
from __future__ import annotations

import logging
from typing import TypedDict

from app.config import Config
from app.services.llm import llm_json, llm_text

logger = logging.getLogger(__name__)


class InterviewAgentState(TypedDict, total=False):
    resume_summary: str
    structured: dict
    job_title: str
    job_requirements: str
    questions: list
    qa_list: list
    current_index: int
    action: str
    user_answer: str
    next_question: str
    score: float
    report: str
    done: bool
    questions_from_llm: bool


def generate_questions(state: InterviewAgentState) -> InterviewAgentState:
    n = Config.INTERVIEW_QUESTION_COUNT
    structured = state.get("structured") or {}
    summary = state.get("resume_summary") or ""
    job_title = (state.get("job_title") or "").strip() or "目标技术岗位"
    job_requirements = (state.get("job_requirements") or "").strip()
    skills = structured.get("skills") or []
    projects = structured.get("projects") or []
    experience = structured.get("experience") or []

    proj_hints = []
    for p in projects[:4]:
        if isinstance(p, dict):
            proj_hints.append(
                f"{p.get('name','')}（{p.get('tech','')}）：{str(p.get('description',''))[:120]}"
            )
        else:
            proj_hints.append(str(p)[:120])

    exp_hints = []
    for e in experience[:3]:
        if isinstance(e, dict):
            exp_hints.append(
                f"{e.get('company','')} {e.get('title','')}：{str(e.get('description',''))[:100]}"
            )
        else:
            exp_hints.append(str(e)[:100])

    prompt = f"""你是资深面试官，正在为岗位「{job_title}」做模拟面试。
必须根据岗位需求 + 候选人简历**现场定制** {n} 道中文问答题，禁止使用千篇一律的通用题。

岗位需求：
{job_requirements or '（未提供详细 JD，请按岗位名称合理出题）'}

硬性要求：
1. 只输出 JSON 数组，长度恰好为 {n}，每个元素是一道完整问题字符串。
2. 至少 2 道题必须点名候选人简历中的具体项目名、公司名或技能（若有）。
3. 题目要能考察候选人与该岗位需求的匹配度。
4. 覆盖：自我介绍与动机、项目深挖、技术基础、场景/故障、学习成长。
5. 不要输出题号、不要 markdown、不要解释。

候选人技能：{skills}
项目线索：{proj_hints or '（结构化项目为空，请从摘要提炼）'}
经历线索：{exp_hints or '（结构化经历为空，请从摘要提炼）'}
简历摘要：
{summary[:2500]}
结构化 JSON：
{structured}
"""

    qs = None
    from_llm = False
    for attempt in range(2):
        data = llm_json(prompt, None, temperature=0.85)
        if isinstance(data, list) and len(data) >= max(3, n - 1):
            qs = [str(q).strip() for q in data if str(q).strip()]
            from_llm = True
            break
        logger.warning("Interview question gen attempt %s failed, retrying", attempt + 1)

    if not qs:
        logger.error("LLM question generation failed; using personalized emergency questions")
        qs = _personalized_emergency(n, skills, proj_hints, summary, job_title)
        from_llm = False

    qs = qs[:n]
    while len(qs) < n:
        qs.append(
            f"结合岗位「{job_title}」与你的经历，请谈谈你在 "
            f"{skills[len(qs) % len(skills)] if skills else '本岗位核心能力'} 方面最有挑战的一次实践。"
        )

    logger.info(
        "Interview questions ready from_llm=%s job=%s count=%s sample=%s",
        from_llm,
        job_title,
        len(qs),
        (qs[0][:60] if qs else ""),
    )
    return {
        **state,
        "questions": qs,
        "qa_list": [],
        "current_index": 0,
        "next_question": qs[0],
        "done": False,
        "action": "generate",
        "questions_from_llm": from_llm,
    }


def _personalized_emergency(
    n: int, skills: list, proj_hints: list, summary: str, job_title: str
) -> list[str]:
    skill = skills[0] if skills else "你的核心技术栈"
    skill2 = skills[1] if len(skills) > 1 else skill
    proj = proj_hints[0] if proj_hints else (summary[:40] or "你最近的项目")
    return [
        f"请结合简历做自我介绍，并说明为何你适合「{job_title}」。",
        f"请深入讲解「{proj}」：你的职责、关键决策与可量化结果。",
        f"你在使用 {skill} 时如何做性能与可靠性取舍？请举简历中的例子。",
        f"若线上出现与 {skill2} 相关的故障，你的排查与沟通流程是什么？",
        f"针对「{job_title}」，你接下来 3 个月计划补强什么？如何验证掌握程度？",
    ][:n]


def evaluate_answer(state: InterviewAgentState) -> InterviewAgentState:
    questions = state.get("questions") or []
    idx = int(state.get("current_index") or 0)
    answer = (state.get("user_answer") or "").strip()
    qa_list = list(state.get("qa_list") or [])

    if idx >= len(questions):
        return {**state, "done": True}

    question = questions[idx]
    fallback = {
        "eval_score": 70.0 if len(answer) > 40 else 50.0,
        "eval_note": "回答已记录。" + ("内容较充实。" if len(answer) > 80 else "建议补充细节。"),
    }
    prompt = f"""你是面试评估官。对候选人回答打分（0-100），输出 JSON：
{{"eval_score": number, "eval_note": "中文点评，指出与题目要求的匹配点与缺口"}}

问题：{question}
回答：{answer}
"""
    data = llm_json(prompt, fallback, temperature=0.2)
    if not isinstance(data, dict):
        data = fallback
    try:
        score = float(data.get("eval_score", fallback["eval_score"]))
    except (TypeError, ValueError):
        score = fallback["eval_score"]
    note = str(data.get("eval_note") or fallback["eval_note"])

    qa_list.append(
        {
            "question": question,
            "answer": answer,
            "eval_score": round(score, 1),
            "eval_note": note,
        }
    )
    next_idx = idx + 1
    done = next_idx >= len(questions)
    next_q = "" if done else questions[next_idx]

    return {
        **state,
        "qa_list": qa_list,
        "current_index": next_idx,
        "next_question": next_q,
        "done": done,
        "action": "evaluate",
        "user_answer": "",
    }


def generate_report(state: InterviewAgentState) -> InterviewAgentState:
    qa_list = state.get("qa_list") or []
    if not qa_list:
        return {**state, "score": 0.0, "report": "无有效问答记录。", "done": True}

    avg = sum(float(x.get("eval_score") or 0) for x in qa_list) / len(qa_list)
    transcript = "\n".join(
        f"Q: {x['question']}\nA: {x['answer']}\n评: {x.get('eval_note')} ({x.get('eval_score')})"
        for x in qa_list
    )
    fallback = (
        f"面试综合分：{avg:.1f}\n"
        f"共完成 {len(qa_list)} 题。整体表现"
        f"{'良好' if avg >= 75 else '中等' if avg >= 60 else '偏弱'}。\n"
        f"建议：结合项目细节与系统设计继续强化表达。\n\n明细：\n{transcript}"
    )
    prompt = f"""根据以下模拟面试记录，写一份中文面试报告，含总分建议、优缺点、录用建议。
平均分约 {avg:.1f}。记录：
{transcript}
"""
    report = llm_text(prompt, fallback)
    return {
        **state,
        "score": round(avg, 1),
        "report": report,
        "done": True,
        "action": "report",
    }

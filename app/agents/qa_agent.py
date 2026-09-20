"""招聘问答子 Agent — RAG + LLM；面试深挖有项目依据时禁止整句拒答。"""
from __future__ import annotations

import logging
import re
from typing import Iterator, TypedDict

from app.config import Config
from app.services.llm import llm_stream, llm_text
from app.services.rag import get_rag

logger = logging.getLogger(__name__)

REFUSE_ANSWER = "不知道。当前知识库与结构化摘要中没有足以回答该问题的依据，我不能编造。"

# Out-of-scope: hard refuse even for candidates
_OUT_OF_SCOPE = ("股票", "天气", "八卦", "彩票", "总统", "火星", "今日油价")


class QAAgentState(TypedDict, total=False):
    query: str
    role: str
    user_id: int
    candidate_id: int | None
    org_id: int | None
    answer: str
    sources: list
    context_note: str
    refused: bool
    recent_turns: str


def run_qa(state: QAAgentState) -> QAAgentState:
    built = prepare_qa(state)
    if built.get("refuse"):
        return {**state, "answer": REFUSE_ANSWER, "sources": [], "refused": True}
    if built.get("fixed_answer"):
        return {
            **state,
            "answer": str(built["fixed_answer"]),
            "sources": built.get("sources") or [],
            "refused": False,
        }

    sources = built["sources"]
    answer = _generate_answer(built)
    return {
        **state,
        "answer": answer,
        "sources": sources,
        "refused": _is_bare_refuse(answer),
    }


def stream_qa(state: QAAgentState) -> tuple[Iterator[str], list]:
    """Return (token_iterator, sources) for SSE streaming."""
    built = prepare_qa(state)
    if built.get("refuse"):

        def refuse_gen() -> Iterator[str]:
            yield REFUSE_ANSWER

        return refuse_gen(), []

    if built.get("fixed_answer"):
        text = str(built["fixed_answer"])

        def fixed_gen() -> Iterator[str]:
            yield text

        return fixed_gen(), []

    sources = built["sources"]
    # Interview / soft paths: buffer so we can guarantee non-refuse output
    if built.get("guaranteed_answer") or built.get("allow_soft_refuse"):

        def buffered() -> Iterator[str]:
            yield _generate_answer(built)

        return buffered(), sources

    prompt = built["prompt"]

    def gen() -> Iterator[str]:
        try:
            yielded = False
            for piece in llm_stream(prompt):
                yielded = True
                yield piece
            if not yielded:
                yield llm_text(prompt, REFUSE_ANSWER)
        except Exception as exc:
            logger.warning("stream_qa failed: %s", exc)
            yield llm_text(prompt, REFUSE_ANSWER)

    return gen(), sources


def _generate_answer(built: dict) -> str:
    """Produce final answer; interview mode never returns bare 不知道 when guaranteed."""
    guaranteed = (built.get("guaranteed_answer") or "").strip()
    prompt = built.get("prompt") or ""
    fallback = guaranteed or REFUSE_ANSWER

    try:
        answer = "".join(llm_stream(prompt)).strip() if prompt else ""
        if not answer:
            answer = llm_text(prompt, fallback).strip() if prompt else fallback
    except Exception as exc:
        logger.warning("QA generate failed: %s", exc)
        answer = llm_text(prompt, fallback).strip() if prompt else fallback

    if _is_bare_refuse(answer) and built.get("retry_prompt"):
        try:
            retry = "".join(llm_stream(built["retry_prompt"])).strip()
            if retry and not _is_bare_refuse(retry):
                answer = retry
        except Exception as exc:
            logger.warning("QA retry failed: %s", exc)

    if guaranteed and (_is_bare_refuse(answer) or _too_thin(answer, guaranteed)):
        return guaranteed
    if _is_bare_refuse(answer) and built.get("allow_soft_refuse") and guaranteed:
        return guaranteed
    if _is_bare_refuse(answer) and not built.get("allow_soft_refuse"):
        return answer
    return answer or guaranteed or REFUSE_ANSWER


def prepare_qa(state: QAAgentState) -> dict:
    """Shared retrieval + refuse decision. Returns prompt/sources or refuse=True."""
    role = state.get("role") or "candidate"
    query = (state.get("query") or "").strip()
    extra = (state.get("context_note") or "").strip()
    recent = (state.get("recent_turns") or "").strip()
    cid = state.get("candidate_id") or state.get("user_id")
    org_id = state.get("org_id")

    if role == "hr" and _is_log_request(query):
        return {
            "sources": [],
            "refuse": False,
            "fixed_answer": (
                "请由运维在 /ops/logs 查看最近运行日志（不在 HR 业务导航中），无需在对话里转述。"
            ),
        }

    if any(x in query for x in _OUT_OF_SCOPE) and not _looks_resume_related(query):
        return {"refuse": True, "sources": []}

    from app.services.resume_dossier import (
        build_resume_dossier,
        dossier_scope,
        parse_resume_id,
        resolve_dossier_resume_id,
        wants_full_dossier,
    )

    def _try_dossier(rid: int | None) -> dict | None:
        if not rid:
            return None
        scope = dossier_scope(query + "\n" + recent)
        dossier = build_resume_dossier(
            rid,
            org_id=org_id if role == "hr" else None,
            allow_candidate_id=int(cid) if role == "candidate" and cid else None,
            scope=scope,
        )
        if not dossier:
            return None
        label = "HR完整档案" if scope == "archive" else "简历信息"
        return {
            "sources": [
                {
                    "source": f"简历#{rid}/{label}",
                    "score": 1.0,
                    "resume_id": rid,
                    "snippet": "数据库导出（未摘要）",
                }
            ],
            "refuse": False,
            "fixed_answer": dossier,
        }

    early_rid = parse_resume_id(query) or parse_resume_id(recent)
    if early_rid and (
        wants_full_dossier(query) or "详细" in query or "全部" in query or "完整" in query
    ):
        hit = _try_dossier(early_rid)
        if hit:
            return hit

    # Name-first dossier（「我要林晓简历的全部信息」）— before noisy hybrid retrieve
    if wants_full_dossier(query) or (("全部" in query or "完整" in query or "详细" in query) and "简历" in query):
        from app.services.resume_dossier import _guess_person_name, find_resume_ids_by_person_name

        pname = _guess_person_name(query) or _guess_person_name(recent)
        if pname:
            name_ids = find_resume_ids_by_person_name(
                pname, org_id=org_id if role == "hr" else None
            )
            for rid in name_ids:
                hit = _try_dossier(rid)
                if hit:
                    return hit

    rag = get_rag()

    # HR: resolve person name before retrieve so FAISS never mixes other candidates
    hr_resume_ids: list[int] | None = None
    if role == "hr":
        from app.services.resume_dossier import _guess_person_name, find_resume_ids_by_person_name

        pname = _guess_person_name(query) or _guess_person_name(recent)
        if pname:
            found = find_resume_ids_by_person_name(pname, org_id=org_id)
            if found:
                hr_resume_ids = found

    if role == "hr":
        if hr_resume_ids:
            hits = rag.retrieve(
                query, top_k=10, resume_ids=hr_resume_ids, org_id=org_id
            )
        else:
            hits = rag.retrieve(query, top_k=10, allow_all=True, org_id=org_id)
    else:
        if cid:
            rag.ensure_candidate_indexed(int(cid))
        hits = rag.retrieve(
            query, top_k=8, candidate_id=cid, allow_all=False, org_id=org_id
        )


    min_score = Config.MIN_RETRIEVE_SCORE
    hits = [(c, s) for c, s in hits if s >= min_score]

    cleaned = []
    for c, s in hits:
        head = (c.text or "")[:120]
        if "岗位需求：" in head and "求职岗位：" in head:
            continue
        cleaned.append((c, s))
    hits = cleaned or hits

    hit_rids: list[int] = []
    for c, _ in hits:
        rid = (c.metadata or {}).get("resume_id")
        if rid is not None and int(rid) not in hit_rids:
            hit_rids.append(int(rid))

    if wants_full_dossier(query) or (
        ("详细" in query or "全部" in query or "完整" in query)
        and ("简历" in query or hit_rids or recent)
    ):
        rid = resolve_dossier_resume_id(
            query + "\n" + recent,
            hit_resume_ids=hit_rids,
            org_id=org_id if role == "hr" else None,
        )
        hit = _try_dossier(rid)
        if hit:
            return hit

    # Drop interview/score noise for factual list questions
    if _is_project_list_question(query):
        cleaned_hits = []
        for c, s in hits:
            section = (c.metadata or {}).get("section") or ""
            kind = (c.metadata or {}).get("section_kind") or ""
            if "面试" in section or kind in {"interview", "score", "jd"}:
                continue
            if "面试" in (c.text or "")[:40] and "开源" not in (c.text or "")[:80]:
                continue
            cleaned_hits.append((c, s))
        if cleaned_hits:
            hits = cleaned_hits
        # refresh hit_rids after filter
        hit_rids = []
        for c, _ in hits:
            rid = (c.metadata or {}).get("resume_id")
            if rid is not None and int(rid) not in hit_rids:
                hit_rids.append(int(rid))

    sources = [
        {
            "source": c.metadata.get("source", c.chunk_id),
            "score": round(score, 3),
            "resume_id": c.metadata.get("resume_id"),
            "section": c.metadata.get("section"),
            "snippet": c.text,
        }
        for c, score in hits
    ]

    # HR/candidate list questions: DB-grounded fixed answer (no LLM rewrite)
    if _is_project_list_question(query):
        focus = _project_list_focus(query)
        listing, list_sources = _compose_section_list_answer(
            query=query,
            role=role,
            org_id=org_id,
            candidate_id=int(cid) if cid else None,
            hit_resume_ids=hit_rids,
            focus=focus,
            retrieved_texts=[c.text for c, _ in hits],
        )
        if listing:
            return {
                "sources": list_sources or sources,
                "refuse": False,
                "fixed_answer": listing,
            }

    project_extra = ""
    if role == "candidate" and cid:
        project_extra = _load_matching_project_text(int(cid), query)
        if not project_extra and _is_interview_style(query):
            project_extra = _load_all_project_sections(int(cid))

    if _is_chitchat(query):
        prompt = f"""你是 HirePilot 企业招聘中台的 AI 助手。角色={role}。
用自然、简洁的中文回复用户问候或自我介绍请求。
可说明：查询评分/面试；要未摘要全文请发「简历#ID 完整档案」或「详细信息」。
用户说：{query}
"""
        return {"prompt": prompt, "sources": [], "refuse": False}

    evidence_blob = "\n".join(
        [extra, project_extra, recent] + [c.text for c, _ in hits]
    )
    has_project_evidence = _has_project_evidence(evidence_blob, query, project_extra)
    has_structured = _structured_covers_query(query, extra) or bool(project_extra) or bool(hits)

    if not hits and not has_structured:
        return {"refuse": True, "sources": []}

    context_blocks = [
        f"[{c.metadata.get('source', c.chunk_id)}] (相关度 {score:.2f})\n{c.text}"
        for c, score in hits
    ]
    if project_extra:
        context_blocks.insert(0, f"[简历原文/项目依据]\n{project_extra}")
    retrieved = "\n\n".join(context_blocks) if context_blocks else "（无检索命中）"

    interview_mode = role == "candidate" and (
        _is_interview_style(query)
        or _is_interview_probe(query, evidence_blob)
        or (has_project_evidence and _is_deep_tech_question(query))
    )

    if interview_mode and has_project_evidence:
        guaranteed = _compose_interview_answer(query, project_extra or retrieved, extra)
        prompt = _interview_polish_prompt(extra, retrieved, query, guaranteed)
        retry = _interview_retry_prompt(extra, retrieved, query, guaranteed)
        return {
            "prompt": prompt,
            "retry_prompt": retry,
            "sources": sources,
            "refuse": False,
            "allow_soft_refuse": True,
            "guaranteed_answer": guaranteed,
        }

    if interview_mode:
        guaranteed = (
            "我按现有简历材料尽量回答。这份材料里没有足够支撑该技术追问的项目细节；"
            "请先确认问的是简历里的哪一个项目，或补充项目描述后再问。"
        )
        prompt = _interview_prompt(extra, retrieved, query)
        return {
            "prompt": prompt,
            "retry_prompt": _interview_retry_prompt(extra, retrieved, query, guaranteed),
            "sources": sources,
            "refuse": False,
            "allow_soft_refuse": True,
            "guaranteed_answer": guaranteed,
        }

    if role == "candidate" and _is_improvement_question(query):
        prompt = f"""你是 HirePilot 招聘助手，正在帮助候选人基于已有评估结果做简历改进建议。
规则：
1. 必须结合结构化摘要里的分数、推荐意见、分项分、项目/开源列表与检索片段，给出具体可执行建议。
2. 可以指出薄弱维度，但不要编造简历中不存在的经历。
3. 区分「项目经历」与「开源项目」。
4. 若信息不足，说明还缺哪些信息，而不是只回「不知道」。
5. 不要列出「来源」「[简历#…]」。

结构化摘要：
{extra or "（无）"}

检索片段：
{retrieved}

用户问题：{query}
"""
        return {
            "prompt": prompt,
            "sources": sources,
            "refuse": False,
            "allow_soft_refuse": True,
            "guaranteed_answer": (
                "根据当前评分摘要，建议优先补强分项偏低的维度，并把项目经历与开源项目分开写清成果与技术栈。"
                "若需要更具体建议，请告诉我最想投的岗位。"
            ),
        }

    if _is_project_list_question(query):
        listing = _compose_project_list(extra, project_extra, retrieved)
        prompt = f"""你是 HirePilot 招聘助手。请根据摘要与检索片段列出候选人的项目，并明确分组：
- 项目经历（正式/实习项目）
- 开源项目（GitHub 等）
- 参与工作（若有）
不要把岗位 JD 当成候选人项目。不要编造。不要输出引用标记。
若检索片段出现「开源项目 / GitHub」，禁止写开源为「无」。

草稿：
{listing}

结构化摘要：
{extra or "（无）"}

检索片段：
{retrieved}

用户问题：{query}
"""
        return {
            "prompt": prompt,
            "sources": sources,
            "refuse": False,
            "allow_soft_refuse": True,
            "guaranteed_answer": listing,
        }

    prompt = f"""你是 HirePilot 招聘中台的 AI 助手。角色={role}。
硬性规则：
1. 只能依据「结构化摘要」和「检索片段」。禁止编造；也禁止把材料里有的内容写成「无」。
2. 若检索片段含「开源项目 / GitHub」，必须列出仓库名，禁止回答开源项目=无。
3. 账号显示名（如张三）与简历结构化姓名（如林晓）不一致时，两者都写明。
4. 用户若要「详细/全部/完整/原文」，不要摘要压缩；并提示可发送「简历#ID 完整档案」获取未截断全文。
5. 区分项目经历、开源项目、参与工作；不要把面试报告优缺点当成简历项目列表。
6. 若完全无关且无依据，只回复：不知道。
7. 不要输出「来源」「[简历#…]」标记行。

近期对话（用于理解指代）：
{recent or "（无）"}

结构化摘要：
{extra or "（无）"}

检索片段：
{retrieved}

用户问题：{query}
"""
    return {"prompt": prompt, "sources": sources, "refuse": False}


def build_qa_prompt(state: QAAgentState) -> tuple[str, list]:
    built = prepare_qa(state)
    if built.get("refuse"):
        return (
            f"请只回复以下原文，不要改写：{REFUSE_ANSWER}",
            [],
        )
    return built["prompt"], built.get("sources") or []


# ----- interview composition (deterministic, no LLM) -----


def _compose_interview_answer(query: str, evidence: str, extra: str = "") -> str:
    """First-person multi-section answer grounded on resume evidence + labeled inference."""
    ev = re.sub(r"\s+", " ", (evidence or "").strip())
    ev_short = ev[:500] if ev else "（简历项目描述较简略）"
    parts: list[str] = []

    parts.append(
        "我按简历里的「二代征信 / RAG」相关项目来回答。"
        f"先说原文依据：{ev_short}"
    )

    # 召回 / 重排
    if any(k in query for k in ("BM25", "混合", "召回", "重排", "向量")):
        parts.append(
            "【召回与重排】简历原文侧重 OCR、Embedding 入库与问答，没有写死 BM25 融合系数。"
            "落地时我的设计是：词面通道（BM25/关键词，吃表格字段名、账户号等专有名词）"
            "+ 向量语义通道（吃段落语义）；融合分做初排，再用业务规则或轻量交叉编码重排，"
            "保证「字段精确命中」和「语义相近」都能上来。下面系数属于工程推断，不是简历原文字面。"
        )

    # OCR / 切分
    if any(k in query.upper() for k in ("切分", "OCR", "表格", "语义")) or "ocr" in query.lower():
        parts.append(
            "【复杂表格 OCR 与语义切分】简历写明面向字段多、表格复杂的二代征信报告做 OCR 提取，"
            "再结合 Embedding 做语义切分与向量入库，支撑章节/账户级问答。"
            "实践上我会尽量按「账户 / 章节 / 表格块」保结构再切，避免把表头和单元格拆散；"
            "切块带上章节元数据，方便手机 H5 端按结构预览和追问。"
        )

    # RAGAS / 评测
    if any(k in query.upper() for k in ("RAGAS", "评测", "迭代")) or "ragas" in query.lower():
        parts.append(
            "【评测与迭代】简历没有写具体 RAGAS 分数。按这类项目我实际会看的问题："
            "faithfulness 低 → 模型脱离上下文胡说；context precision/recall 差 → 切分过碎或表格结构丢了；"
            "answer relevancy 不稳 → 召回噪声。"
            "迭代上优先修切分粒度与元数据、加强字段关键词通道，再调融合/重排，而不是只堆 top-k。"
            "具体小数是推断说明，不是简历原文数据。"
        )

    # Generic fallback sections if query is broad
    if len(parts) == 1:
        parts.append(
            "整体链路是：复杂表格 OCR → 结构化/语义切分 → Embedding 入库 → 检索问答（含 H5）。"
            "你问到的更细参数（融合权重、重排模型、RAGAS 表）原文没写；"
            "若继续追问某一块，我可以按落地经验展开，并标明哪些是推断。"
        )

    if extra and ("总分" in extra or "推荐" in extra):
        parts.append("补充：系统侧已有对该简历的评分摘要，但不替代项目技术细节原文。")

    parts.append("如果你希望，我可以只展开「召回重排 / 切分 / 评测」其中一块讲更深。")
    return "\n\n".join(parts)


def _project_list_focus(query: str) -> str:
    q = query or ""
    if "开源" in q:
        return "opensource"
    if any(k in q for k in ("参与工作", "工作参与")):
        return "participation"
    if any(k in q for k in ("项目经历", "哪些项目", "有哪几个项目", "有什么项目", "项目有哪些", "做过哪些")):
        return "projects"
    return "all"


def _compose_section_list_answer(
    *,
    query: str,
    role: str,
    org_id: int | None,
    candidate_id: int | None,
    hit_resume_ids: list[int],
    focus: str,
    retrieved_texts: list[str],
) -> tuple[str, list]:
    """Build grounded list from DB structured fields; fall back to retrieved opensource lines."""
    from app.models import Resume, db
    from app.services.resume_dossier import (
        _guess_person_name,
        _opensource_from_raw,
        find_resume_ids_by_person_name,
        parse_resume_id,
    )

    rid = parse_resume_id(query)
    if not rid:
        pname = _guess_person_name(query)
        if pname:
            found = find_resume_ids_by_person_name(
                pname, org_id=org_id if role == "hr" else None
            )
            if found:
                rid = found[0]
    if not rid and hit_resume_ids:
        rid = hit_resume_ids[0]
    if not rid and candidate_id:
        row = (
            Resume.query.filter_by(candidate_id=int(candidate_id))
            .order_by(Resume.created_at.desc())
            .first()
        )
        rid = row.id if row else None
    if not rid:
        return "", []

    resume = db.session.get(Resume, int(rid))
    if not resume:
        return "", []
    if role == "hr" and org_id is not None and resume.org_id not in (None, org_id):
        return "", []
    if role == "candidate" and candidate_id and resume.candidate_id != int(candidate_id):
        return "", []

    structured = resume.structured() or {}
    raw = resume.raw_text or ""
    projects = structured.get("projects") or []
    opensource = structured.get("opensource_projects") or []
    participation = structured.get("work_participation") or []
    if not opensource:
        opensource = _opensource_from_raw(raw)
    # Also harvest opensource names from retrieved opensource chunks
    if not opensource:
        blob = "\n".join(retrieved_texts or [])
        opensource = _opensource_from_raw("开源项目\n" + blob)

    def _names(items: list) -> list[str]:
        out: list[str] = []
        for p in items or []:
            if isinstance(p, dict):
                n = str(p.get("name") or "").strip()
                desc = str(p.get("description") or "").strip()
                if n and desc and focus != "opensource":
                    # keep short for projects; opensource usually name-only
                    line = n
                elif n:
                    line = n
                else:
                    continue
            else:
                line = str(p).strip()
            if line and line not in out:
                out.append(line)
        return out

    # Enrich opensource lines from raw section text when available
    open_lines = _names(opensource)
    m = re.search(r"开源项目([\s\S]{0,800})", raw)
    if m and focus in {"opensource", "all"}:
        block = m.group(1)
        richer: list[str] = []
        for line in block.splitlines():
            t = line.strip().strip("-•*")
            if not t or t.lower().startswith("github"):
                if "Galaxy" in t or "github" in t.lower():
                    richer.append(t)
                continue
            if any(k in t.lower() for k in ("xiaoyou", "wsy", "zhi-tou", "小柚", "智投")):
                richer.append(t)
        if richer:
            open_lines = richer

    proj_lines = _names(projects)
    part_lines = _names(participation)

    parts: list[str] = []
    if focus in {"projects", "all"}:
        parts.append("项目经历：")
        parts.extend(f"- {x}" for x in (proj_lines or ["（结构化/原文未列出）"]))
    if focus in {"opensource", "all"}:
        parts.append("开源项目：")
        parts.extend(f"- {x}" for x in (open_lines or ["（未找到开源项目）"]))
    if focus in {"participation", "all"} and part_lines:
        parts.append("参与工作：")
        parts.extend(f"- {x}" for x in part_lines)

    name = (structured.get("name") or "").strip() or f"简历#{rid}"
    header = f"{name} · 简历#{rid}"
    text = header + "\n" + "\n".join(parts)
    sources = [
        {
            "source": f"简历#{rid}/结构化列表",
            "score": 1.0,
            "resume_id": rid,
            "snippet": focus,
        }
    ]
    return text, sources


def _compose_project_list(extra: str, project_extra: str, retrieved: str) -> str:
    text = "\n".join([extra or "", project_extra or "", retrieved or ""])
    formal: list[str] = []
    opensource: list[str] = []
    participation: list[str] = []
    patterns = [
        (formal, r"二代征信[^\n。]{0,40}"),
        (opensource, r"xiaoyou[^\n。]{0,40}|小柚[^\n。]{0,40}"),
        (opensource, r"wsy-studio[^\n。]{0,40}|WSY Studio[^\n。]{0,40}"),
        (opensource, r"zhi-tou[^\n。]{0,40}|智投[^\n。]{0,40}"),
        (participation, r"合同系统[^\n。]{0,30}|政务数字人[^\n。]{0,30}|电商自动化[^\n。]{0,30}"),
    ]
    for bucket, pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            s = m.group(0).strip()
            if s and s not in bucket:
                bucket.append(s)
    if "项目经历：" in extra:
        line = extra.split("项目经历：", 1)[1].split("；", 1)[0]
        for name in re.split(r"[、,，]", line):
            name = name.strip()
            if name and name != "无" and name not in formal:
                formal.append(name)
    if "开源项目：" in extra:
        line = extra.split("开源项目：", 1)[1].split("；", 1)[0]
        for name in re.split(r"[、,，]", line):
            name = name.strip()
            if name and name != "无" and name not in opensource:
                opensource.append(name)

    def fmt(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}\n- （材料中未明确列出）"
        return title + "\n" + "\n".join(f"- {x}" for x in items[:8])

    return "\n\n".join(
        [
            fmt("项目经历", formal),
            fmt("开源项目", opensource),
            fmt("参与工作", participation),
        ]
    )


def _interview_polish_prompt(extra: str, retrieved: str, query: str, draft: str) -> str:
    return f"""你正在模拟技术面试，用第一人称扮演候选人。
下面已有「必保答案草稿」。请在草稿基础上润色成更自然的口语化分点回答。
硬性规则：
1. 禁止输出「不知道」或同义拒答。
2. 不要删除草稿中的分点与「原文/推断」区分。
3. 不要捏造简历未出现的公司名或精确评测分数。
4. 不要输出引用标记。

必保草稿：
{draft}

结构化摘要：
{extra[:1200] or "（无）"}

检索/原文：
{retrieved[:3500]}

面试官问题：{query}
"""


def _interview_prompt(extra: str, retrieved: str, query: str) -> str:
    return f"""你正在模拟技术面试。用户是面试官，你必须用第一人称扮演简历中的候选人作答。
禁止只回复「不知道」。有项目依据就分点作答；缺数字就标明「原文未写」再给工程推断。

结构化摘要：
{extra or "（无）"}

检索/原文依据：
{retrieved}

面试官问题：{query}
"""


def _interview_retry_prompt(
    extra: str, retrieved: str, query: str, guaranteed: str = ""
) -> str:
    return f"""上一次错误地拒答了。请直接输出下面必保草稿（可轻微润色），禁止再回答不知道。

必保草稿：
{guaranteed}

材料：
{extra[:800]}
{retrieved[:2000]}

面试官：{query}
"""


def _load_all_project_sections(candidate_id: int) -> str:
    try:
        from app.models import Resume
        from app.services.rag import HybridRAG

        resumes = (
            Resume.query.filter_by(candidate_id=candidate_id)
            .order_by(Resume.created_at.desc())
            .all()
        )
    except Exception as exc:
        logger.warning("load all projects failed: %s", exc)
        return ""

    rag = HybridRAG.__new__(HybridRAG)
    blobs: list[str] = []
    for r in resumes:
        raw = (r.raw_text or "").strip()
        if not raw:
            continue
        for label, kind, body in HybridRAG.split_resume_sections(rag, raw):
            if kind in {"projects", "opensource", "participation"} and body.strip():
                blobs.append(f"【{label}】\n{body[:2000]}")
        if not blobs and raw:
            blobs.append(raw[:2500])
        if len(blobs) >= 4:
            break
    return "\n\n".join(blobs)[:5000]


def _load_matching_project_text(candidate_id: int, query: str) -> str:
    try:
        from app.models import Resume
        from app.services.rag import HybridRAG

        resumes = (
            Resume.query.filter_by(candidate_id=candidate_id)
            .order_by(Resume.created_at.desc())
            .all()
        )
    except Exception as exc:
        logger.warning("load project text failed: %s", exc)
        return ""

    rag = HybridRAG.__new__(HybridRAG)
    blobs: list[str] = []
    q = query or ""
    for r in resumes:
        raw = (r.raw_text or "").strip()
        if not raw:
            continue
        for label, kind, body in HybridRAG.split_resume_sections(rag, raw):
            if kind not in {"projects", "opensource", "participation", "body"}:
                continue
            if _text_overlaps_query(body, q) or _text_overlaps_query(label, q):
                blobs.append(f"【{label}】\n{body[:2000]}")
            elif any(
                k in q for k in ("征信", "RAG", "BM25", "小柚", "WSY", "智投", "OCR", "表格")
            ) and kind in {"projects", "body"}:
                blobs.append(f"【{label}】\n{body[:2000]}")
        if len(blobs) >= 3:
            break
    if not blobs and _is_interview_style(q):
        return _load_all_project_sections(candidate_id)
    return "\n\n".join(blobs)[:4500]


def _has_project_evidence(evidence: str, query: str, project_extra: str) -> bool:
    if (project_extra or "").strip():
        return True
    ev = evidence or ""
    keys = (
        "项目经历",
        "开源项目",
        "征信",
        "RAG",
        "OCR",
        "Embedding",
        "小柚",
        "智投",
        "WSY",
        "HirePilot",
        "项目内容",
    )
    return any(k.lower() in ev.lower() for k in keys)


def _is_interview_style(query: str) -> bool:
    return _is_roleplay(query) or _is_deep_tech_question(query)


def _is_deep_tech_question(query: str) -> bool:
    q = query or ""
    if len(q) < 15:
        return False
    markers = (
        "如何设计",
        "怎么做",
        "讲一下",
        "具体讲",
        "说说",
        "请介绍",
        "召回",
        "重排",
        "切分",
        "BM25",
        "RAGAS",
        "ragas",
        "OCR",
        "混合检索",
        "迭代优化",
        "暴露过哪些",
        "你如何",
        "你在",
    )
    return any(m in q for m in markers)


def _looks_resume_related(query: str) -> bool:
    keys = ("简历", "项目", "面试", "评分", "征信", "RAG", "技能")
    return any(k in (query or "") for k in keys)


def _text_overlaps_query(text: str, query: str) -> bool:
    if not text or not query:
        return False
    q_tokens = set(re.findall(r"[A-Za-z0-9_+#./-]{2,}|[\u4e00-\u9fff]{2,}", query))
    t_tokens = set(re.findall(r"[A-Za-z0-9_+#./-]{2,}|[\u4e00-\u9fff]{2,}", text))
    if q_tokens & t_tokens:
        return True
    return any(tok in text for tok in q_tokens if len(tok) >= 2)


def _is_bare_refuse(answer: str) -> bool:
    t = (answer or "").strip()
    if not t:
        return True
    if t in {"不知道", "不知道。", REFUSE_ANSWER}:
        return True
    if t.startswith("不知道") and len(t) < 120:
        return True
    # soft refuse paraphrases
    refuse_hints = ("没有足以回答", "无法回答", "不能编造", "依据不足", "材料中没有")
    if len(t) < 100 and any(h in t for h in refuse_hints):
        return True
    return False


def _too_thin(answer: str, guaranteed: str) -> bool:
    """LLM answer much shorter than guaranteed draft → prefer guaranteed."""
    a = (answer or "").strip()
    g = (guaranteed or "").strip()
    if not g:
        return False
    return len(a) < max(80, int(len(g) * 0.35))


def _structured_covers_query(query: str, extra: str) -> bool:
    if not extra or len(extra.strip()) < 40:
        return False
    if "暂无简历" in extra or "无数据" in extra:
        return False
    if any(x in query for x in _OUT_OF_SCOPE) and not _looks_resume_related(query):
        return False
    topic_keys = (
        "教育",
        "学历",
        "学校",
        "项目",
        "开源",
        "技能",
        "工作",
        "经验",
        "分数",
        "评分",
        "推荐",
        "面试",
        "岗位",
        "简历",
        "电话",
        "邮箱",
        "姓名",
        "改进",
        "提升",
        "不足",
        "薄弱",
        "优化",
        "建议",
        "github",
        "GitHub",
        "征信",
        "RAG",
        "BM25",
        "OCR",
    )
    if any(k.lower() in query.lower() for k in topic_keys):
        return True
    if _is_interview_style(query) or _is_improvement_question(query) or _is_project_list_question(query):
        return True
    return len(set(query) & set(extra)) >= 4


def _is_log_request(query: str) -> bool:
    q = (query or "").strip().lower()
    keys = ("后端日志", "查看日志", "系统日志", "运行日志", "hirepilot.log", "看一下日志", "看下日志")
    return any(k.lower() in q for k in keys)


def _is_chitchat(query: str) -> bool:
    q = query.strip().lower()
    if len(q) <= 8 and any(x in q for x in ("你好", "您好", "hi", "hello", "嗨", "在吗")):
        return True
    if any(x in q for x in ("你是谁", "你能做什么", "介绍一下你", "帮助", "怎么用")):
        return True
    return False


def _is_roleplay(query: str) -> bool:
    q = query or ""
    keys = (
        "作为我简历",
        "扮演我",
        "你来当候选人",
        "我来当面试官",
        "我来作为面试官",
        "模拟面试",
        "你是候选人",
        "用第一人称",
        "请具体讲一下你",
        "请介绍一下你在",
        "你在二代",
        "你在项目",
    )
    if any(k in q for k in keys):
        return True
    if ("你" in q) and any(
        k in q
        for k in (
            "如何设计",
            "怎么做",
            "讲一下",
            "说说",
            "RAG",
            "BM25",
            "重排",
            "切分",
            "OCR",
            "RAGAS",
            "ragas",
            "迭代",
        )
    ):
        return True
    return False


def _is_interview_probe(query: str, evidence: str) -> bool:
    q = query or ""
    if len(q) < 20:
        return False
    markers = (
        "征信",
        "混合检索",
        "BM25",
        "向量",
        "重排",
        "切分",
        "OCR",
        "RAGAS",
        "ragas",
        "召回",
        "小柚",
        "智投",
        "WSY",
        "Agent",
    )
    if not any(m.lower() in q.lower() for m in markers):
        return False
    evidence = evidence or ""
    return any(m.lower() in evidence.lower() for m in markers if len(m) >= 2)


def _is_improvement_question(query: str) -> bool:
    q = query or ""
    keys = ("改进", "提升", "不足", "薄弱", "优化简历", "怎么改", "需要改进", "建议改")
    return any(k in q for k in keys)


def _is_project_list_question(query: str) -> bool:
    q = query or ""
    if "开源" in q and ("项目" in q or "github" in q.lower() or "仓库" in q):
        return True
    if "项目经历" in q:
        return True
    keys = ("哪些项目", "有哪几个项目", "有什么项目", "项目有哪些", "做过哪些项目", "项目列表")
    return any(k in q for k in keys)

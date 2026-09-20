"""Full resume dossier dump for HR/candidate detail requests (no LLM summarization)."""
from __future__ import annotations

import json
import re

from app.services.weights import get_rank_weights


def parse_resume_id(query: str) -> int | None:
    q = query or ""
    patterns = (
        r"简历\s*#\s*(\d+)",
        r"简历\s*[编号]*\s*(\d+)",
        r"#\s*(\d+)",
        r"resume[_\s#-]*(\d+)",
    )
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if m:
            return int(m.group(1))
    # bare number only when query is essentially just the id
    m = re.fullmatch(r"\s*(\d{1,6})\s*", q)
    if m:
        return int(m.group(1))
    return None


def wants_full_dossier(query: str) -> bool:
    q = query or ""
    keys = (
        "详细信息",
        "全部信息",
        "完整信息",
        "完整原文",
        "简历原文",
        "不要总结",
        "不是总结",
        "不是你总结",
        "全部内容",
        "完整内容",
        "原始简历",
        "全文",
        "完整字段",
        "完整档案",
        ".dump",
        "dossier",
    )
    if any(k in q for k in keys):
        return True
    if "详细" in q and any(k in q for k in ("简历", "信息", "这个", "#", "完整")):
        return True
    # 「我要林晓简历的全部信息」
    if ("全部" in q or "完整" in q or "详细" in q) and "简历" in q:
        return True
    return False


def dossier_scope(query: str) -> str:
    """resume = 仅简历正文；archive = 含 JD/面试/评分明细。"""
    q = query or ""
    if any(
        k in q
        for k in (
            "完整档案",
            "含面试",
            "面试记录",
            "面试报告",
            "评分明细",
            "岗位需求",
            "全部档案",
            ".dump",
            "dossier",
        )
    ):
        return "archive"
    return "resume"


def find_resume_ids_by_person_name(
    name: str,
    *,
    org_id: int | None = None,
) -> list[int]:
    """Match structured resume name first; account display_name only if no conflicting struct name."""
    name = (name or "").strip()
    if not name:
        return []
    from app.models import Resume

    q = Resume.query
    if org_id is not None:
        q = q.filter_by(org_id=org_id)
    rows = q.order_by(Resume.created_at.desc()).limit(80).all()
    exact_struct: list[int] = []
    soft_struct: list[int] = []
    display_only: list[int] = []
    raw_only: list[int] = []
    for r in rows:
        cand = r.candidate
        struct = r.structured() or {}
        struct_name = str(struct.get("name") or "").strip()
        display = ((cand.display_name if cand else "") or "").strip()
        raw_head = (r.raw_text or "")[:120]

        if struct_name and (name == struct_name or struct_name.startswith(name)):
            exact_struct.append(r.id)
            continue
        if struct_name and name in struct_name:
            soft_struct.append(r.id)
            continue
        # Account display match only when structured name is empty or same as display
        # (avoids: 账号张三 + 简历林晓 → 问「张三」误命中林晓)
        if display and (name == display or name in display):
            if not struct_name or struct_name == display:
                display_only.append(r.id)
            continue
        if name in raw_head and (not struct_name or name in struct_name or name == display):
            raw_only.append(r.id)

    for bucket in (exact_struct, soft_struct, display_only, raw_only):
        if bucket:
            out: list[int] = []
            for i in bucket:
                if i not in out:
                    out.append(i)
            return out
    return []


def build_resume_dossier(
    resume_id: int,
    *,
    org_id: int | None = None,
    allow_candidate_id: int | None = None,
    scope: str = "resume",
) -> str | None:
    """Export resume text. Default scope=resume (no JD / interview / score dump)."""
    from app.models import Resume, db

    resume = db.session.get(Resume, resume_id)
    if not resume:
        return None
    if org_id is not None and resume.org_id not in (None, org_id):
        return None
    if allow_candidate_id is not None and resume.candidate_id != allow_candidate_id:
        return None

    cand = resume.candidate
    display = cand.display_name if cand else "?"
    uid = resume.candidate_id
    structured = resume.structured() or {}
    archive = (scope or "resume").strip().lower() == "archive"

    lines: list[str] = []
    title = "HR完整档案" if archive else "简历信息"
    lines.append(f"======== 简历#{resume.id} {title}（未摘要） ========")
    struct_name = (structured.get("name") or "").strip()
    subject = struct_name or display or "?"
    lines.append(f"姓名：{subject}")
    if display and struct_name and display != struct_name:
        lines.append(f"上传账号：{display}（uid={uid}；系统账号，非另一位候选人）")
    if resume.target_job_title:
        lines.append(f"求职岗位：{resume.target_job_title}")

    # Readable structured sections (not raw JSON wall)
    lines.append("")
    lines.extend(_format_structured_sections(structured, resume.raw_text or ""))

    lines.append("")
    lines.append("-------- 简历原文 --------")
    lines.append((resume.raw_text or "").strip() or "（无原文）")

    if archive:
        resume_w, interview_w = get_rank_weights(resume.org_id)
        comp = resume.composite_score(resume_w, interview_w)
        best = resume.best_interview_score()
        lines.append("")
        lines.append("-------- 岗位需求原文 --------")
        lines.append((resume.target_job_requirements or "").strip() or "（无）")
        lines.append("")
        lines.append("-------- 评分摘要 --------")
        lines.append(
            f"总分={resume.total_score} 技能={resume.skill_score} 项目={resume.project_score} "
            f"教育={resume.education_score} 契合={resume.fit_score} 综合分={round(comp, 1)} "
            f"推荐={resume.recommendation} 达标={resume.meets_threshold} "
            f"最高面试分={best if best is not None else '无'}"
        )
        if resume.recommendation_reason:
            lines.append(f"推荐理由：{resume.recommendation_reason}")

        interviews = [
            s
            for s in (
                resume.interviews.all()
                if hasattr(resume.interviews, "all")
                else resume.interviews
            )
            if s.status == "completed"
        ]
        interviews = sorted(interviews, key=lambda s: s.round_no or 0)
        lines.append("")
        lines.append("-------- 面试记录 --------")
        if not interviews:
            lines.append("（暂无已完成面试）")
        for s in interviews:
            lines.append(f"### 面试第{s.round_no}场 · 得分={s.score}")
            qa = s.qa_list() or []
            if qa:
                lines.append("问答明细：")
                for i, item in enumerate(qa, 1):
                    if isinstance(item, dict):
                        lines.append(f"Q{i}: {item.get('q') or item.get('question') or ''}")
                        lines.append(f"A{i}: {item.get('a') or item.get('answer') or ''}")
                        ev = item.get("eval") or item.get("score")
                        if ev is not None:
                            lines.append(f"评{i}: {ev}")
                    else:
                        lines.append(str(item))
            if s.report:
                lines.append("面试报告全文：")
                lines.append(s.report)
            lines.append("")

        details = resume.score_details() if hasattr(resume, "score_details") else {}
        if details:
            lines.append("-------- 评分明细 JSON --------")
            lines.append(json.dumps(details, ensure_ascii=False, indent=2))
    else:
        lines.append("")
        lines.append("（仅简历信息。若要面试记录/评分明细/岗位JD，请说「完整档案」。）")

    lines.append("======== 结束 ========")
    return "\n".join(lines)


def _format_structured_sections(structured: dict, raw_text: str) -> list[str]:
    """Compact facts only — full prose lives in 简历原文."""
    lines: list[str] = ["-------- 要点 --------"]
    email = structured.get("email") or ""
    phone = structured.get("phone") or ""
    if email or phone:
        lines.append(f"联系方式：邮箱={email or '—'}  电话={phone or '—'}")

    edu = structured.get("education") or []
    if edu:
        bits = []
        for e in edu:
            if isinstance(e, dict):
                bits.append(
                    f"{e.get('period') or ''} {e.get('school') or ''} "
                    f"{e.get('degree') or ''} {e.get('major') or ''}".strip()
                )
            else:
                bits.append(str(e))
        lines.append("教育：" + "；".join(b for b in bits if b))

    exp = structured.get("experience") or []
    if exp:
        bits = []
        for x in exp:
            if isinstance(x, dict):
                bits.append(
                    f"{x.get('period') or ''} {x.get('company') or ''} "
                    f"{x.get('title') or ''}".strip()
                )
            else:
                bits.append(str(x))
        lines.append("经历：" + "；".join(b for b in bits if b))

    skills = structured.get("skills") or []
    if skills:
        # one line each is long; join short previews
        short = []
        for s in skills:
            t = str(s).strip()
            short.append(t if len(t) <= 40 else t[:38] + "…")
        lines.append("技能：" + "；".join(short))

    projects = structured.get("projects") or []
    opensource = structured.get("opensource_projects") or []
    participation = structured.get("work_participation") or []
    if not opensource:
        opensource = _opensource_from_raw(raw_text)

    lines.append("项目经历：" + _names(projects))
    lines.append("开源项目：" + _names(opensource))
    if participation:
        lines.append("参与工作：" + _names(participation))
    return lines


def resolve_dossier_resume_id(
    query: str,
    *,
    hit_resume_ids: list[int] | None = None,
    org_id: int | None = None,
) -> int | None:
    rid = parse_resume_id(query)
    if rid:
        return rid

    # Prefer explicit person name over noisy retrieval top-1 (avoids 张三串入林晓)
    name = _guess_person_name(query)
    if name:
        found = find_resume_ids_by_person_name(name, org_id=org_id)
        if found:
            # If retrieval also hit some of these, prefer intersection order
            if hit_resume_ids:
                for hid in hit_resume_ids:
                    if hid in found:
                        return hid
            return found[0]

    if wants_full_dossier(query) and hit_resume_ids:
        return hit_resume_ids[0]
    return None


def _guess_person_name(query: str) -> str:
    q = query or ""

    def _clean(name: str) -> str:
        name = (name or "").strip()
        for prefix in ("要", "看", "查", "问", "给", "的", "求"):
            if name.startswith(prefix) and len(name) > 2:
                name = name[len(prefix) :]
        return name

    # Prefer explicit「我要林晓简历…」
    m = re.search(
        r"(?:我要|查看|给出|导出|请问)?\s*([\u4e00-\u9fff]{2,4})\s*[·•]?\s*简历",
        q,
    )
    if m:
        return _clean(m.group(1))
    m = re.search(
        r"([\u4e00-\u9fff]{2,4})\s*(的)?(简历)?(的)?(详细|全部|完整|信息)",
        q,
    )
    if m:
        return _clean(m.group(1))
    return ""


def _names(items: list) -> str:
    out = []
    for p in items or []:
        if isinstance(p, dict):
            n = str(p.get("name") or "").strip()
            if n:
                out.append(n)
        else:
            s = str(p).strip()
            if s:
                out.append(s)
    return "、".join(out) if out else "（结构化字段为空——请以下方原文为准）"


def _opensource_from_raw(raw: str) -> list[dict]:
    """Extract GitHub / 开源项目 names from raw resume text."""
    if not raw or "开源" not in raw:
        return []
    # Take from 开源项目 header to next major section / end
    m = re.search(r"开源项目([\s\S]{0,1200})", raw)
    if not m:
        return []
    block = m.group(1)
    names: list[dict] = []
    for pat in (
        r"(xiaoyou[\w-]*)",
        r"(wsy[\w-]*)",
        r"(zhi-tou[\w-]*)",
        r"([A-Za-z][\w-]{2,40})\s*[：:]",
    ):
        for hit in re.finditer(pat, block, flags=re.I):
            n = hit.group(1).strip()
            if n.lower() in {"github", "http", "https"}:
                continue
            if not any(d.get("name", "").lower() == n.lower() for d in names):
                names.append({"name": n})
    return names

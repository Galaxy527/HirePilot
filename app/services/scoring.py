"""Resume structuring and multi-dimension scoring (简历筛选子 Agent logic)."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from app.config import Config
from app.services.llm import llm_json
from app.services.weights import DEFAULT_DIM, normalize_dims, weighted_total

logger = logging.getLogger(__name__)

TARGET_SKILLS = [
    "Python",
    "Flask",
    "Django",
    "FastAPI",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "Redis",
    "Docker",
    "Linux",
    "Git",
    "REST",
    "API",
    "算法",
    "LangChain",
]


@dataclass
class DimensionDetail:
    name: str
    score: float
    weight: float
    comment: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    total_score: float
    skill_score: float
    project_score: float
    education_score: float
    fit_score: float
    recommendation: str
    recommendation_reason: str
    structured: dict
    details: dict = field(default_factory=dict)


JOB_PROFILE = """
目标岗位：全栈 / 后端工程师（Python / Web）。
关注：Python、Flask/Django、SQL、算法基础、项目落地、沟通协作。
"""


def resolve_job_profile(job_title: str | None = None, job_requirements: str | None = None) -> str:
    title = (job_title or "").strip()
    req = (job_requirements or "").strip()
    if title or req:
        return f"目标岗位：{title or '未命名岗位'}\n岗位需求：\n{req or '（未提供详细需求，请按岗位名称合理推断）'}"
    return JOB_PROFILE.strip()


# Equal weight for four dimensions (default; org may override)
DIM_WEIGHTS = dict(DEFAULT_DIM)


def structure_resume(raw_text: str) -> dict:
    fallback = _heuristic_structure(raw_text)
    prompt = f"""你是简历解析助手。将以下简历清洗为 JSON，字段：
name, email, phone, education(list of {{school,degree,major,period}}),
skills(list of str),
projects(list of {{name,description,tech}}) — 仅「项目经历/项目经验」中的正式项目，不要把 GitHub 开源仓库混进来,
opensource_projects(list of {{name,description,tech,repo}}) — 「开源项目/GitHub」下列出的仓库,
work_participation(list of {{name,description}}) — 参与过但非独立负责的工作/模块（如合同系统、数字人等）,
experience(list of {{company,title,period,description}}), summary(str)。
只输出 JSON。务必区分 projects 与 opensource_projects。

简历原文：
{raw_text[:8000]}
"""
    data = llm_json(prompt, fallback)
    if not isinstance(data, dict):
        return fallback
    for key in ("name", "email", "phone", "summary"):
        data.setdefault(key, fallback.get(key, ""))
    for key in ("education", "skills", "projects", "opensource_projects", "work_participation", "experience"):
        data.setdefault(key, fallback.get(key, []))
    return data


def score_resume(
    raw_text: str,
    structured: dict | None = None,
    *,
    job_title: str | None = None,
    job_requirements: str | None = None,
    dim_weights: dict[str, float] | None = None,
) -> ScoreResult:
    structured = structured or structure_resume(raw_text)
    job_profile = resolve_job_profile(job_title, job_requirements)
    weights = normalize_dims(dim_weights or DIM_WEIGHTS)
    fallback = _heuristic_score(
        raw_text, structured, job_profile=job_profile, dim_weights=weights
    )
    prompt = f"""你是企业招聘简历筛选专家。必须严格对照下列「企业岗位需求」评估候选人匹配度，而不是用通用标准。

企业岗位需求：
{job_profile}

请对候选人打分（0-100），输出 JSON（务必填写详细评语，不要只给分数）：
{{
  "skill_score": number,
  "project_score": number,
  "education_score": number,
  "fit_score": number,
  "total_score": number,
  "recommendation": "推荐"|"待定"|"不推荐",
  "recommendation_reason": "2-4句总体评价，说明相对该岗位为何给出该推荐意见",
  "skill_comment": "相对该岗位的技能评语（含命中与缺口）",
  "project_comment": "相对该岗位的项目经验评语",
  "education_comment": "教育背景评语",
  "fit_comment": "与该岗位职责/要求的契合度评语",
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["不足1", "不足2"],
  "matched_skills": ["已匹配该岗位的技能"],
  "missing_skills": ["该岗位期望但简历未体现的技能"],
  "highlights": ["与该岗位相关的简历亮点"]
}}
总分由系统按企业权重加权计算，你仍可给出参考 total_score。结构化简历：
{structured}
原文摘要：
{raw_text[:4000]}
只输出 JSON。
"""
    data = llm_json(prompt, _result_to_llm_fallback(fallback), temperature=0.2)
    if not isinstance(data, dict):
        return fallback

    def f(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    skill = f("skill_score", fallback.skill_score)
    project = f("project_score", fallback.project_score)
    education = f("education_score", fallback.education_score)
    fit = f("fit_score", fallback.fit_score)
    total = weighted_total(skill, project, education, fit, weights)
    rec = str(data.get("recommendation") or fallback.recommendation)
    if rec not in ("推荐", "待定", "不推荐"):
        rec = _rec_from_score(total)
    reason = str(data.get("recommendation_reason") or fallback.recommendation_reason)

    details = _merge_details(
        scores={
            "skill": skill,
            "project": project,
            "education": education,
            "fit": fit,
            "total": total,
        },
        llm_data=data,
        fallback_details=fallback.details,
        structured=structured,
        raw_text=raw_text,
        job_profile=job_profile,
        dim_weights=weights,
    )

    return ScoreResult(
        total_score=round(total, 1),
        skill_score=round(skill, 1),
        project_score=round(project, 1),
        education_score=round(education, 1),
        fit_score=round(fit, 1),
        recommendation=rec,
        recommendation_reason=reason,
        structured=structured,
        details=details,
    )


def build_details_for_resume(resume) -> dict:
    """Reconstruct / load score details for display (supports legacy rows)."""
    stored = resume.score_details() if hasattr(resume, "score_details") else {}
    if stored and stored.get("dimensions"):
        # ensure job profile shown
        if not stored.get("job_profile") and hasattr(resume, "job_profile_text"):
            stored = {**stored, "job_profile": resume.job_profile_text()}
        return stored
    structured = resume.structured() if hasattr(resume, "structured") else {}
    raw = resume.raw_text or ""
    job_profile = resume.job_profile_text() if hasattr(resume, "job_profile_text") else JOB_PROFILE
    result = _heuristic_score(
        raw, structured or _heuristic_structure(raw), job_profile=job_profile
    )
    if resume.skill_score is not None:
        result.details["dimensions"]["skill"]["score"] = resume.skill_score
    if resume.project_score is not None:
        result.details["dimensions"]["project"]["score"] = resume.project_score
    if resume.education_score is not None:
        result.details["dimensions"]["education"]["score"] = resume.education_score
    if resume.fit_score is not None:
        result.details["dimensions"]["fit"]["score"] = resume.fit_score
    if resume.total_score is not None:
        result.details["summary"]["total_score"] = resume.total_score
    if resume.recommendation:
        result.details["summary"]["recommendation"] = resume.recommendation
    if resume.recommendation_reason:
        result.details["summary"]["recommendation_reason"] = resume.recommendation_reason
    return result.details


def _rec_from_score(total: float) -> str:
    if total >= 80:
        return "推荐"
    if total >= Config.RESUME_SCORE_THRESHOLD:
        return "待定"
    return "不推荐"


def _heuristic_structure(raw_text: str) -> dict:
    email_m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw_text)
    phone_m = re.search(r"1[3-9]\d{9}", raw_text)
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    name = lines[0][:40] if lines else "未知"
    skills_kw = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "Flask",
        "Django",
        "FastAPI",
        "React",
        "Vue",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "Redis",
        "Docker",
        "Linux",
        "Git",
        "LangChain",
        "机器学习",
        "算法",
        "REST",
        "API",
    ]
    found = [s for s in skills_kw if s.lower() in raw_text.lower() or s in raw_text]
    return {
        "name": name,
        "email": email_m.group(0) if email_m else "",
        "phone": phone_m.group(0) if phone_m else "",
        "education": [],
        "skills": found,
        "projects": [],
        "opensource_projects": [],
        "work_participation": [],
        "experience": [],
        "summary": raw_text[:300],
    }


def _matched_and_missing(
    raw_text: str, structured: dict, job_profile: str | None = None
) -> tuple[list[str], list[str]]:
    text_blob = (raw_text + " " + " ".join(structured.get("skills") or [])).lower()
    # Prefer skills mentioned in job profile; fall back to default target list
    profile = job_profile or ""
    profile_skills = []
    for s in TARGET_SKILLS:
        if s.lower() in profile.lower() or s in profile:
            profile_skills.append(s)
    # also pull capitalized tokens / common tech from profile
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9+.#]{1,20}|[\u4e00-\u9fff]{2,12}", profile):
        if tok not in profile_skills and len(tok) >= 2:
            profile_skills.append(tok)
    pool = profile_skills[:20] if profile_skills else TARGET_SKILLS

    matched = []
    missing = []
    for skill in pool:
        if skill.lower() in text_blob or skill in (structured.get("skills") or []):
            matched.append(skill)
        else:
            missing.append(skill)
    if "API" in matched and "REST" in missing:
        missing = [m for m in missing if m != "REST"]
    return matched, missing[:12]


def _level_label(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 75:
        return "良好"
    if score >= 65:
        return "一般"
    if score >= 50:
        return "偏弱"
    return "不足"


def _heuristic_score(
    raw_text: str,
    structured: dict,
    job_profile: str | None = None,
    dim_weights: dict[str, float] | None = None,
) -> ScoreResult:
    job_profile = job_profile or JOB_PROFILE.strip()
    skills = list(structured.get("skills") or [])
    text_l = raw_text.lower()
    matched, missing = _matched_and_missing(raw_text, structured, job_profile=job_profile)
    skill_hits = len(matched) or len(skills)
    skill_score = min(95.0, 45 + skill_hits * 5.5)

    projects = structured.get("projects") or []
    project_hints = len(re.findall(r"项目|project|github", text_l))
    project_score = min(92.0, 50 + project_hints * 8 + len(projects) * 10)

    edu_list = structured.get("education") or []
    edu_hints = len(re.findall(r"本科|硕士|博士|大学|bachelor|master", text_l))
    education_score = min(90.0, 55 + edu_hints * 10 + min(10, len(edu_list) * 5))

    # Extract keywords from job profile for fit
    fit_tokens = [t for t in re.findall(r"[a-zA-Z+#.]{2,}|[\u4e00-\u9fff]{2,}", job_profile)]
    fit_hits = []
    for tok in fit_tokens:
        tl = tok.lower()
        if tl in text_l or tok in raw_text:
            if tok not in fit_hits and len(fit_hits) < 12:
                fit_hits.append(tok)
    # also classic keywords if profile empty-ish
    if len(fit_hits) < 2:
        for k in ["flask", "django", "python", "sql", "api", "后端", "全栈", "fastapi"]:
            if k in text_l:
                fit_hits.append(k)
    fit_score = min(95.0, 40 + len(fit_hits) * 6)

    weights = normalize_dims(dim_weights or DIM_WEIGHTS)
    total = weighted_total(skill_score, project_score, education_score, fit_score, weights)
    rec = _rec_from_score(total)

    skill_comment = (
        f"相对岗位「{job_profile.split(chr(10))[0]}」技能匹配{_level_label(skill_score)}"
        f"（{round(skill_score, 1)} 分）。命中 {len(matched)} 项"
        + (f"：{('、'.join(matched[:8]))}" if matched else "")
        + "。"
        + (f"缺口：{('、'.join(missing[:6]))}。" if missing else "")
    )
    if projects:
        proj_names = [
            str(p.get("name") or p)[:40] for p in projects[:3] if isinstance(p, dict) or True
        ]
        project_comment = (
            f"项目经验{_level_label(project_score)}（{round(project_score, 1)} 分）。"
            f"解析到 {len(projects)} 个项目"
            + (f"（如 {'、'.join(str(n) for n in proj_names if n)}）" if proj_names else "")
            + "，请对照岗位需求看相关性。"
        )
    else:
        project_comment = (
            f"项目经验{_level_label(project_score)}（{round(project_score, 1)} 分）。"
            f"正文中项目相关表述约 {project_hints} 处；结构化项目列表为空。"
        )

    education_comment = (
        f"教育背景{_level_label(education_score)}（{round(education_score, 1)} 分）。"
        + (
            f"识别到学历/院校相关信息 {edu_hints} 处。"
            if edu_hints
            else "学历信息不够清晰。"
        )
    )
    fit_comment = (
        f"岗位契合{_level_label(fit_score)}（{round(fit_score, 1)} 分）。"
        f"与岗位需求关键词重合 {len(fit_hits)} 个"
        + (f"（{('、'.join(fit_hits[:8]))}）" if fit_hits else "")
        + "。"
    )

    strengths = []
    if matched:
        strengths.append(f"掌握 {('、'.join(matched[:5]))} 等与该岗位相关技能")
    if project_score >= 70:
        strengths.append("具备可验证的项目/工程实践表述")
    if fit_score >= 70:
        strengths.append("经历方向与目标岗位较契合")
    if not strengths:
        strengths.append("简历已提供基础信息，具备继续完善的空间")

    weaknesses = []
    if missing[:4]:
        weaknesses.append(f"相对该岗位仍缺：{('、'.join(missing[:4]))}")
    if project_score < 70:
        weaknesses.append("项目描述偏少或与岗位关联不强")
    if education_score < 70:
        weaknesses.append("教育背景信息不够完整")
    if fit_score < Config.RESUME_SCORE_THRESHOLD:
        weaknesses.append("与岗位需求关键词重叠不足")

    highlights = []
    if structured.get("summary"):
        highlights.append(str(structured["summary"])[:120])
    for p in (projects or [])[:2]:
        if isinstance(p, dict):
            highlights.append(f"项目：{p.get('name', '')} — {str(p.get('description', ''))[:80]}")
        else:
            highlights.append(f"项目：{p}")
    if not highlights:
        highlights.append((raw_text[:120] + "…") if len(raw_text) > 120 else raw_text)

    reason = (
        f"对照岗位需求评估，综合分 {total}（阈值 {Config.RESUME_SCORE_THRESHOLD}），结论「{rec}」。"
        f"四维：技能 {round(skill_score,1)} / 项目 {round(project_score,1)} / "
        f"教育 {round(education_score,1)} / 契合 {round(fit_score,1)}。"
        + ("（LLM 未配置时为本地启发式）" if not Config.llm_enabled() else "")
    )

    details = {
        "job_profile": job_profile,
        "weights": weights,
        "dimensions": {
            "skill": {
                "label": "技能匹配",
                "score": round(skill_score, 1),
                "weight": weights["skill"],
                "level": _level_label(skill_score),
                "comment": skill_comment,
                "evidence": matched[:8],
            },
            "project": {
                "label": "项目经验",
                "score": round(project_score, 1),
                "weight": weights["project"],
                "level": _level_label(project_score),
                "comment": project_comment,
                "evidence": [
                    str(p.get("name") if isinstance(p, dict) else p) for p in (projects or [])[:5]
                ],
            },
            "education": {
                "label": "教育背景",
                "score": round(education_score, 1),
                "weight": weights["education"],
                "level": _level_label(education_score),
                "comment": education_comment,
                "evidence": [
                    f"{e.get('school','')} {e.get('degree','')} {e.get('major','')}".strip()
                    if isinstance(e, dict)
                    else str(e)
                    for e in (edu_list or [])[:5]
                ],
            },
            "fit": {
                "label": "岗位契合度",
                "score": round(fit_score, 1),
                "weight": weights["fit"],
                "level": _level_label(fit_score),
                "comment": fit_comment,
                "evidence": fit_hits[:8],
            },
        },
        "matched_skills": matched,
        "missing_skills": missing[:10],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "highlights": [h for h in highlights if h],
        "summary": {
            "total_score": total,
            "recommendation": rec,
            "recommendation_reason": reason,
            "threshold": Config.RESUME_SCORE_THRESHOLD,
        },
    }

    return ScoreResult(
        total_score=total,
        skill_score=round(skill_score, 1),
        project_score=round(project_score, 1),
        education_score=round(education_score, 1),
        fit_score=round(fit_score, 1),
        recommendation=rec,
        recommendation_reason=reason,
        structured=structured,
        details=details,
    )


def _result_to_llm_fallback(result: ScoreResult) -> dict:
    d = asdict(result)
    # flatten a bit for llm_json fallback compatibility
    details = result.details or {}
    dims = details.get("dimensions") or {}
    return {
        "skill_score": result.skill_score,
        "project_score": result.project_score,
        "education_score": result.education_score,
        "fit_score": result.fit_score,
        "total_score": result.total_score,
        "recommendation": result.recommendation,
        "recommendation_reason": result.recommendation_reason,
        "skill_comment": (dims.get("skill") or {}).get("comment", ""),
        "project_comment": (dims.get("project") or {}).get("comment", ""),
        "education_comment": (dims.get("education") or {}).get("comment", ""),
        "fit_comment": (dims.get("fit") or {}).get("comment", ""),
        "strengths": details.get("strengths") or [],
        "weaknesses": details.get("weaknesses") or [],
        "matched_skills": details.get("matched_skills") or [],
        "missing_skills": details.get("missing_skills") or [],
        "highlights": details.get("highlights") or [],
        "structured": d.get("structured"),
        "details": details,
    }


def _as_str_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def _merge_details(
    *,
    scores: dict,
    llm_data: dict,
    fallback_details: dict,
    structured: dict,
    raw_text: str,
    job_profile: str | None = None,
    dim_weights: dict[str, float] | None = None,
) -> dict:
    base = dict(fallback_details or {})
    dims = dict(base.get("dimensions") or {})
    weights = normalize_dims(dim_weights or DIM_WEIGHTS)
    mapping = {
        "skill": ("skill_comment", scores["skill"]),
        "project": ("project_comment", scores["project"]),
        "education": ("education_comment", scores["education"]),
        "fit": ("fit_comment", scores["fit"]),
    }
    labels = {
        "skill": "技能匹配",
        "project": "项目经验",
        "education": "教育背景",
        "fit": "岗位契合度",
    }
    for key, (comment_key, score) in mapping.items():
        prev = dict(dims.get(key) or {})
        comment = str(llm_data.get(comment_key) or prev.get("comment") or "")
        dims[key] = {
            "label": labels[key],
            "score": round(score, 1),
            "weight": weights[key],
            "level": _level_label(score),
            "comment": comment,
            "evidence": prev.get("evidence") or [],
        }

    matched = _as_str_list(llm_data.get("matched_skills")) or base.get("matched_skills") or []
    missing = _as_str_list(llm_data.get("missing_skills")) or base.get("missing_skills") or []
    if not matched and not missing:
        matched, missing = _matched_and_missing(
            raw_text, structured, job_profile=job_profile
        )

    return {
        "job_profile": (job_profile or base.get("job_profile") or JOB_PROFILE).strip(),
        "weights": weights,
        "dimensions": dims,
        "matched_skills": matched,
        "missing_skills": missing,
        "strengths": _as_str_list(llm_data.get("strengths")) or base.get("strengths") or [],
        "weaknesses": _as_str_list(llm_data.get("weaknesses")) or base.get("weaknesses") or [],
        "highlights": _as_str_list(llm_data.get("highlights")) or base.get("highlights") or [],
        "summary": {
            "total_score": round(scores["total"], 1),
            "recommendation": str(llm_data.get("recommendation") or ""),
            "recommendation_reason": str(llm_data.get("recommendation_reason") or ""),
            "threshold": Config.RESUME_SCORE_THRESHOLD,
        },
    }

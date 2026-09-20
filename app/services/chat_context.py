"""Build short DB context strings for Agent chat (role-scoped)."""
from __future__ import annotations

import json

from app.config import Config
from app.models import Resume
from app.services.weights import get_rank_weights


def _fmt_project_names(items: list, limit: int = 6) -> str:
    names: list[str] = []
    for p in items or []:
        if isinstance(p, dict):
            n = str(p.get("name") or "").strip()
        else:
            n = str(p).strip()
        if n:
            names.append(n)
        if len(names) >= limit:
            break
    return "、".join(names) if names else "无"


def candidate_context_note(user_id: int) -> str:
    resumes = (
        Resume.query.filter_by(candidate_id=user_id)
        .order_by(Resume.created_at.desc())
        .all()
    )
    if not resumes:
        return "该候选人尚未上传简历。"
    lines = [f"候选人本人数据摘要（共 {len(resumes)} 份简历）："]
    for r in resumes:
        best = r.best_interview_score()
        lines.append(
            f"- 简历#{r.id} 岗位={r.target_job_title or '未指定'} 文件={r.filename or '文本'} 总分={r.total_score} "
            f"推荐={r.recommendation} 达标={r.meets_threshold} "
            f"技能={r.skill_score}/项目={r.project_score}/教育={r.education_score}/契合={r.fit_score} "
            f"面试次数={r.interview_count()}/{Config.MAX_INTERVIEWS_PER_RESUME} "
            f"最高面试分={best if best is not None else '无'} "
            f"理由={((r.recommendation_reason or '')[:120])}"
        )
        structured = r.structured() or {}
        if structured:
            projects = structured.get("projects") or []
            opensource = structured.get("opensource_projects") or []
            participation = structured.get("work_participation") or []
            lines.append(
                "  项目经历："
                + _fmt_project_names(projects)
                + "；开源项目："
                + _fmt_project_names(opensource)
                + "；参与工作："
                + _fmt_project_names(participation)
            )
            lines.append(
                "  结构化："
                + json.dumps(structured, ensure_ascii=False)[:1200]
            )
            # Lightweight improvement hints from score gaps
            hints = []
            for label, val in (
                ("技能", r.skill_score),
                ("项目", r.project_score),
                ("教育", r.education_score),
                ("岗位契合", r.fit_score),
            ):
                if val is not None and float(val) < 80:
                    hints.append(f"{label}分偏低({val})")
            if hints:
                lines.append("  可改进线索：" + "；".join(hints))
        raw = (r.raw_text or "").strip()
        if raw:
            lines.append(f"  简历原文摘要：{raw[:1200]}")
    return "\n".join(lines)


def hr_context_note(org_id: int | None = None) -> str:
    threshold = Config.RESUME_SCORE_THRESHOLD
    resume_w, interview_w = get_rank_weights(org_id)
    q = Resume.query
    if org_id is not None:
        q = q.filter_by(org_id=org_id)
    all_resumes = q.order_by(Resume.total_score.desc()).limit(30).all()
    qualified = [r for r in all_resumes if r.meets_threshold]
    lines = [
        f"HR 业务摘要（组织={org_id}）：阈值={threshold}；"
        f"综合排序权重 简历={resume_w:.2f}/面试={interview_w:.2f}；"
        f"当前展示样本 {len(all_resumes)} 份，其中达标 {len(qualified)} 份。",
        "候选人列表（按简历分，最多30）：",
    ]
    for r in all_resumes:
        name = r.candidate.display_name if r.candidate else "?"
        best = r.best_interview_score()
        comp = r.composite_score(resume_w, interview_w)
        structured = r.structured() or {}
        lines.append(
            f"- {name}(uid={r.candidate_id}) 简历#{r.id} 岗位={r.target_job_title or '未指定'} "
            f"简历分={r.total_score} 综合分={round(comp,1)} 推荐={r.recommendation} "
            f"达标={r.meets_threshold} 最高面试分={best if best is not None else '无'} "
            f"项目={_fmt_project_names(structured.get('projects') or [], 3)} "
            f"开源={_fmt_project_names(structured.get('opensource_projects') or [], 3)}"
        )
        raw = (r.raw_text or "").strip()
        if raw:
            lines.append(f"  原文摘要：{raw[:280]}")
    return "\n".join(lines)

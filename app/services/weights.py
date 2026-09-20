"""Organization-level scoring / ranking weights (HR configurable)."""
from __future__ import annotations

from typing import Any

from app.config import Config
from app.models import Organization, Resume, db

DIM_KEYS = ("skill", "project", "education", "fit")
DEFAULT_DIM = {
    "skill": 0.25,
    "project": 0.25,
    "education": 0.25,
    "fit": 0.25,
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def normalize_pair(a: float, b: float) -> tuple[float, float]:
    a, b = max(0.0, float(a)), max(0.0, float(b))
    s = a + b
    if s <= 0:
        return Config.RANK_RESUME_WEIGHT, Config.RANK_INTERVIEW_WEIGHT
    return a / s, b / s


def normalize_dims(weights: dict[str, float]) -> dict[str, float]:
    raw = {k: max(0.0, float(weights.get(k, DEFAULT_DIM[k]))) for k in DIM_KEYS}
    s = sum(raw.values())
    if s <= 0:
        return dict(DEFAULT_DIM)
    return {k: round(v / s, 6) for k, v in raw.items()}


def weighted_total(
    skill: float,
    project: float,
    education: float,
    fit: float,
    dim_weights: dict[str, float] | None = None,
) -> float:
    w = normalize_dims(dim_weights or DEFAULT_DIM)
    total = (
        skill * w["skill"]
        + project * w["project"]
        + education * w["education"]
        + fit * w["fit"]
    )
    return round(total, 1)


def get_org(org_id: int | None) -> Organization | None:
    if not org_id:
        return None
    return db.session.get(Organization, org_id)


def get_rank_weights(org_id: int | None) -> tuple[float, float]:
    org = get_org(org_id)
    if not org:
        return Config.RANK_RESUME_WEIGHT, Config.RANK_INTERVIEW_WEIGHT
    return normalize_pair(
        org.rank_resume_weight if org.rank_resume_weight is not None else Config.RANK_RESUME_WEIGHT,
        org.rank_interview_weight
        if org.rank_interview_weight is not None
        else Config.RANK_INTERVIEW_WEIGHT,
    )


def get_dim_weights(org_id: int | None) -> dict[str, float]:
    org = get_org(org_id)
    if not org:
        return dict(DEFAULT_DIM)
    return normalize_dims(
        {
            "skill": org.weight_skill if org.weight_skill is not None else 0.25,
            "project": org.weight_project if org.weight_project is not None else 0.25,
            "education": org.weight_education if org.weight_education is not None else 0.25,
            "fit": org.weight_fit if org.weight_fit is not None else 0.25,
        }
    )


def save_weights(
    org: Organization,
    *,
    rank_resume: float,
    rank_interview: float,
    dim: dict[str, float],
) -> dict[str, Any]:
    rw, iw = normalize_pair(rank_resume, rank_interview)
    dims = normalize_dims(dim)
    org.rank_resume_weight = rw
    org.rank_interview_weight = iw
    org.weight_skill = dims["skill"]
    org.weight_project = dims["project"]
    org.weight_education = dims["education"]
    org.weight_fit = dims["fit"]
    return {"rank_resume": rw, "rank_interview": iw, "dims": dims}


def recompute_org_totals(org_id: int) -> int:
    """Recompute total_score / threshold from stored dimension scores using current dim weights."""
    dims = get_dim_weights(org_id)
    threshold = Config.RESUME_SCORE_THRESHOLD
    resumes = Resume.query.filter_by(org_id=org_id).filter(Resume.processing_status == "ready").all()
    n = 0
    for r in resumes:
        if None in (r.skill_score, r.project_score, r.education_score, r.fit_score):
            continue
        total = weighted_total(
            r.skill_score, r.project_score, r.education_score, r.fit_score, dims
        )
        r.total_score = total
        r.meets_threshold = total >= threshold
        details = r.score_details() if hasattr(r, "score_details") else None
        if isinstance(details, dict):
            details = dict(details)
            details["weights"] = dims
            dimensions = dict(details.get("dimensions") or {})
            for key in DIM_KEYS:
                if key in dimensions and isinstance(dimensions[key], dict):
                    dimensions[key] = {**dimensions[key], "weight": dims[key]}
            details["dimensions"] = dimensions
            summary = dict(details.get("summary") or {})
            summary["total_score"] = total
            summary["threshold"] = threshold
            details["summary"] = summary
            r.set_score_details(details)
        n += 1
    return n


def parse_percent(value: str | None, default_pct: float) -> float:
    """Accept 0-100 percent input; return 0-1 weight."""
    if value is None or str(value).strip() == "":
        return default_pct / 100.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default_pct / 100.0
    if v > 1.0:
        v = v / 100.0
    return _clamp01(v)

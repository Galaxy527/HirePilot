"""Scoring contract against golden fixtures (heuristic / mock LLM)."""
import json
from pathlib import Path

from app.services.scoring import score_resume


def test_scoring_contract(app):
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures" / "golden_resumes.json").read_text(encoding="utf-8")
    )
    with app.app_context():
        for item in fixtures:
            result = score_resume(
                item["resume_text"],
                job_title=item["job_title"],
                job_requirements=item["job_requirements"],
            )
            assert result.recommendation in item["recommendation_in"]
            if "min_total" in item:
                assert result.total_score >= item["min_total"]
            if "max_total" in item:
                assert result.total_score <= item["max_total"]

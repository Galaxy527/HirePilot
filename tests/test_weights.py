"""Org weight save / recompute / scoring."""
from app.models import Organization, Resume, User, db
from app.services.scoring import score_resume
from app.services.weights import get_dim_weights, get_rank_weights


def test_weighted_total_respects_dim_weights(app):
    with app.app_context():
        result = score_resume(
            "熟悉 Python Flask SQL Docker，有后端项目经验，本科计算机。",
            job_title="Python 后端",
            job_requirements="Python Flask SQL Docker",
            dim_weights={"skill": 0.7, "project": 0.1, "education": 0.1, "fit": 0.1},
        )
        expected = round(
            result.skill_score * 0.7
            + result.project_score * 0.1
            + result.education_score * 0.1
            + result.fit_score * 0.1,
            1,
        )
        assert result.total_score == expected
        assert abs(result.details["weights"]["skill"] - 0.7) < 1e-6


def test_hr_can_save_weights_and_recompute(hr_client, app):
    with app.app_context():
        org = Organization.query.filter_by(code="DEMO").first()
        assert org is not None
        cand = User.query.filter_by(username="candidate1").first()
        resume = Resume(
            candidate_id=cand.id,
            org_id=org.id,
            raw_text="demo",
            processing_status="ready",
            skill_score=80,
            project_score=60,
            education_score=40,
            fit_score=20,
            total_score=50,
            meets_threshold=False,
        )
        db.session.add(resume)
        db.session.commit()
        rid = resume.id

    rv = hr_client.post(
        "/hr/settings/weights",
        data={
            "rank_resume": "60",
            "rank_interview": "40",
            "weight_skill": "40",
            "weight_project": "30",
            "weight_education": "20",
            "weight_fit": "10",
        },
        follow_redirects=True,
    )
    assert rv.status_code == 200
    assert "权重已保存" in rv.get_data(as_text=True)

    with app.app_context():
        org = Organization.query.filter_by(code="DEMO").first()
        rw, iw = get_rank_weights(org.id)
        assert abs(rw - 0.6) < 1e-6
        assert abs(iw - 0.4) < 1e-6
        dims = get_dim_weights(org.id)
        assert abs(dims["skill"] - 0.4) < 1e-6
        r = db.session.get(Resume, rid)
        # 80*0.4 + 60*0.3 + 40*0.2 + 20*0.1 = 60
        assert r.total_score == 60.0

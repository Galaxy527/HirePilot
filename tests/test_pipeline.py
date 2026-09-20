"""Pipeline stage + notifications."""


def test_reject_requires_reason(hr_client, app):
    with app.app_context():
        from app.models import JobPosting, Resume, User, db

        cand = User.query.filter_by(username="candidate1").first()
        job = JobPosting.query.first()
        r = Resume(
            candidate_id=cand.id,
            org_id=cand.org_id,
            raw_text="test",
            processing_status="ready",
            pipeline_stage="screening",
            meets_threshold=True,
            total_score=80,
            job_id=job.id,
            target_job_title=job.title,
            target_job_requirements=job.requirements,
        )
        db.session.add(r)
        db.session.commit()
        rid = r.id

    rv = hr_client.post(
        f"/hr/resume/{rid}",
        data={"action": "pipeline", "pipeline_stage": "rejected", "reject_reason": ""},
        follow_redirects=True,
    )
    assert "淘汰必须选择原因" in rv.get_data(as_text=True)

    rv = hr_client.post(
        f"/hr/resume/{rid}",
        data={
            "action": "pipeline",
            "pipeline_stage": "rejected",
            "reject_reason": "skill_mismatch",
            "reject_note": "技能不够",
        },
        follow_redirects=True,
    )
    assert rv.status_code == 200
    with app.app_context():
        from app.models import Notification, Resume

        r = Resume.query.get(rid)
        assert r.pipeline_stage == "rejected"
        assert r.reject_reason == "skill_mismatch"
        notes = Notification.query.filter_by(user_id=r.candidate_id).all()
        assert any("状态更新" in n.title for n in notes)

"""Audit log coverage."""


def test_hr_view_writes_audit(hr_client, app):
    with app.app_context():
        from app.models import JobPosting, Resume, User, db

        cand = User.query.filter_by(username="candidate1").first()
        job = JobPosting.query.first()
        r = Resume(
            candidate_id=cand.id,
            raw_text="audit resume",
            processing_status="ready",
            pipeline_stage="screening",
            meets_threshold=True,
            total_score=88,
            recommendation="推荐",
            job_id=job.id,
            target_job_title=job.title,
            target_job_requirements=job.requirements,
        )
        db.session.add(r)
        db.session.commit()
        rid = r.id

    rv = hr_client.get(f"/hr/resume/{rid}")
    assert rv.status_code == 200
    with app.app_context():
        from app.models import AuditLog

        rows = AuditLog.query.filter_by(action="hr_view_resume", entity_id=str(rid)).all()
        assert len(rows) >= 1

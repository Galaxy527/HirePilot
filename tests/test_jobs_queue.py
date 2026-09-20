"""Background job queue with inline execution."""


def test_enqueue_score_resume_inline(app):
    with app.app_context():
        from app.models import JobPosting, Resume, User, db
        from app.services import jobs as jobq
        from app.services.resume_service import create_resume_draft

        user = User.query.filter_by(username="candidate1").first()
        job = JobPosting.query.first()
        resume = create_resume_draft(
            candidate_id=user.id,
            raw_text="张三，Python Flask SQL 项目经验丰富，本科学历。",
            filename="t.txt",
            job_id=job.id,
            job_title=job.title,
            job_requirements=job.requirements,
        )
        assert resume.processing_status == "queued"
        bg = jobq.enqueue(
            jobq.KIND_SCORE_RESUME,
            {"resume_id": resume.id, "candidate_name": user.display_name},
        )
        db.session.refresh(resume)
        db.session.refresh(bg)
        assert bg.status == "succeeded"
        assert resume.processing_status == "ready"
        assert resume.total_score is not None

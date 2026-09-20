"""Resume dossier dump helpers."""
from app.services.resume_dossier import parse_resume_id, wants_full_dossier


def test_parse_resume_id():
    assert parse_resume_id("张三 · 简历 #2我要这个详细信息") == 2
    assert parse_resume_id("简历#2 完整档案") == 2
    assert parse_resume_id("请给出简历 2 的全部信息") == 2
    assert parse_resume_id("2") == 2


def test_wants_full_dossier():
    assert wants_full_dossier("我需要全部信息不是你总结的")
    assert wants_full_dossier("张三 · 简历 #2我要这个详细信息")
    assert wants_full_dossier("完整原文")
    assert wants_full_dossier("我要林晓简历的全部信息")
    assert not wants_full_dossier("他有几份简历")


def test_guess_maolin_name():
    from app.services.resume_dossier import _guess_person_name

    assert _guess_person_name("我要林晓简历的全部信息") == "林晓"
    assert _guess_person_name("张三 · 简历 #2") == "张三"
    assert "要" not in _guess_person_name("我要林晓简历的全部信息")


def test_dossier_prepare_fixed(app):
    with app.app_context():
        from app.agents import qa_agent as qa
        from app.models import Resume, User, db
        from app.services.tenant import get_or_create_default_org

        org = get_or_create_default_org()
        u = User.query.filter_by(username="candidate1").first()
        assert u
        r = Resume(
            candidate_id=u.id,
            org_id=org.id,
            raw_text=(
                "林晓\n开源项目\nGitHub：Galaxy527 xiaoyou-chat-agent\n"
                "项目经历\n二代征信报告识别与RAG问答系统"
            ),
            target_job_title="AI应用开发工程师",
            total_score=88,
            recommendation="推荐",
            meets_threshold=True,
            processing_status="ready",
        )
        r.set_structured(
            {
                "name": "林晓",
                "projects": [{"name": "二代征信报告识别与RAG问答系统"}],
                "opensource_projects": [{"name": "xiaoyou-chat-agent"}],
            }
        )
        db.session.add(r)
        db.session.commit()
        rid = r.id

        built = qa.prepare_qa(
            {
                "query": f"简历#{rid} 我要这个详细信息，全部信息不要总结",
                "role": "hr",
                "user_id": 1,
                "org_id": org.id,
                "context_note": "",
                "recent_turns": "",
            }
        )
        assert built.get("fixed_answer")
        text = built["fixed_answer"]
        assert "完整档案" in text
        assert "林晓" in text
        assert "xiaoyou" in text or "开源" in text
        assert "二代征信" in text
        # cleanup
        db.session.delete(r)
        db.session.commit()

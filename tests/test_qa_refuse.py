"""QA hard refuse."""


def test_refuse_without_evidence(app):
    with app.app_context():
        from app.agents.qa_agent import REFUSE_ANSWER, run_qa

        out = run_qa(
            {
                "query": "今天火星上的天气如何？",
                "role": "candidate",
                "user_id": 1,
                "candidate_id": 1,
                "context_note": "",
            }
        )
        assert out.get("refused") or out["answer"].startswith("不知道")
        assert "编造" in REFUSE_ANSWER or out["answer"].startswith("不知道")

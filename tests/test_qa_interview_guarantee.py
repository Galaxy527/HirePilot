"""Interview deep-dive must not bare-refuse when project evidence exists."""
from app.agents import qa_agent as qa


SAMPLE_Q = (
    "你在二代征信报告识别与RAG问答系统中用了BM25和向量检索做混合检索，"
    "请具体讲一下你如何设计召回与重排策略、如何处理复杂表格OCR后的语义切分，"
    "以及RAGAS评测结果暴露过哪些问题、你如何迭代优化。"
)

PROJECT_EV = (
    "【项目经历】\n"
    "二代征信报告识别与 RAG 问答系统："
    "面向二代个人征信报告字段多、表格复杂，实现复杂表格 OCR 提取，"
    "并结合 Embedding 完成语义切分与向量入库，支撑报告结构化预览、章节/账户级问答与手机 H5。"
)


def test_compose_covers_all_facets():
    ans = qa._compose_interview_answer(SAMPLE_Q, PROJECT_EV, "")
    assert "不知道" not in ans
    assert "召回" in ans or "重排" in ans
    assert "切分" in ans or "OCR" in ans
    assert "评测" in ans or "RAGAS" in ans or "faithfulness" in ans
    assert len(ans) > 200


def test_prepare_qa_interview_has_guaranteed(app):
    with app.app_context():
        built = qa.prepare_qa(
            {
                "query": SAMPLE_Q,
                "role": "candidate",
                "user_id": 99999,
                "candidate_id": 99999,
                "context_note": (
                    "候选人本人数据摘要（共 1 份简历）：\n"
                    "- 简历#9 总分=88 推荐=推荐 项目经历：二代征信报告识别与RAG问答系统\n"
                    f"  简历原文摘要：{PROJECT_EV}"
                ),
            }
        )
        assert not built.get("refuse")
        guaranteed = built.get("guaranteed_answer") or ""
        assert guaranteed
        assert "不知道" not in guaranteed
        assert built.get("allow_soft_refuse") is True


def test_generate_answer_falls_back_to_guaranteed(monkeypatch):
    guaranteed = qa._compose_interview_answer(SAMPLE_Q, PROJECT_EV, "")

    def fake_stream(_prompt):
        yield "不知道。"

    monkeypatch.setattr(qa, "llm_stream", fake_stream)
    monkeypatch.setattr(qa, "llm_text", lambda *a, **k: "不知道。")

    out = qa._generate_answer(
        {
            "prompt": "x",
            "retry_prompt": "y",
            "guaranteed_answer": guaranteed,
            "allow_soft_refuse": True,
        }
    )
    assert not out.startswith("不知道")
    assert "召回" in out or "OCR" in out


def test_mars_still_refused(app):
    with app.app_context():
        out = qa.run_qa(
            {
                "query": "今天火星上的天气如何？",
                "role": "candidate",
                "user_id": 1,
                "candidate_id": 1,
                "context_note": "",
            }
        )
        assert out.get("refused") or out["answer"].startswith("不知道")

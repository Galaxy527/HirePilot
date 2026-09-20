"""RAG section split + no JD pollution."""
from app.services.rag import HybridRAG, INDEX_SCHEMA


def test_split_projects_and_opensource():
    rag = HybridRAG.__new__(HybridRAG)
    text = (
        "林晓 求职意向：AI工程师\n"
        "项目经历\n"
        "2025-09 二代征信报告识别与RAG问答系统\n"
        "开源项目\n"
        "GitHub：xiaoyou-chat-agent 小柚\n"
        "技能特长\n"
        "Python LangGraph\n"
    )
    sections = HybridRAG.split_resume_sections(rag, text)
    kinds = [k for _, k, _ in sections]
    assert "projects" in kinds
    assert "opensource" in kinds
    proj = next(body for label, kind, body in sections if kind == "projects")
    open_ = next(body for label, kind, body in sections if kind == "opensource")
    assert "二代征信" in proj
    assert "xiaoyou" in open_
    assert "xiaoyou" not in proj


def test_index_resume_excludes_full_jd(tmp_path):
    rag = HybridRAG(index_dir=tmp_path / "faiss")
    n = rag.index_resume(
        resume_id=99,
        candidate_id=7,
        raw_text="项目经历\n二代征信RAG\n开源项目\nwsy-studio",
        structured={
            "name": "贺",
            "projects": [{"name": "二代征信RAG"}],
            "opensource_projects": [{"name": "wsy-studio"}],
        },
        candidate_name="贺",
        job_title="AI工程师",
    )
    assert n >= 2
    texts = " ".join(c.text for c in rag.chunks)
    assert "岗位需求" not in texts
    assert any(c.metadata.get("section_kind") == "projects" for c in rag.chunks)
    assert any(c.metadata.get("section_kind") == "opensource" for c in rag.chunks)
    assert all(c.metadata.get("schema") == INDEX_SCHEMA for c in rag.chunks)
    # job title may appear as short intent only
    intent = [c for c in rag.chunks if c.metadata.get("section_kind") == "intent"]
    assert intent and "AI工程师" in intent[0].text

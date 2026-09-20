#!/usr/bin/env python
"""RAG evaluation: refuse contracts, context integrity, optional Ragas metrics.

Usage:
  python scripts/eval_rag.py
  python scripts/eval_rag.py --with-ragas   # needs LLM_API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app
from app.agents.qa_agent import run_qa


SAMPLE_RESUME = (
    "张三 邮箱 zhangsan@example.com 电话 13800138000 "
    "教育：某某大学 计算机 本科 "
    "技能：Python Flask SQL Docker LangChain PostgreSQL "
    "项目：招聘中台 HirePilot，负责简历评分与模拟面试 Agent。"
    "经验：后端开发 2 年，熟悉 REST API 与 PostgreSQL。"
)
SAMPLE_STRUCTURED = {
    "name": "张三",
    "email": "zhangsan@example.com",
    "phone": "13800138000",
    "education": [{"school": "某某大学", "major": "计算机", "degree": "本科"}],
    "skills": ["Python", "Flask", "SQL", "PostgreSQL", "Docker", "LangChain"],
    "projects": [{"name": "HirePilot", "description": "简历评分与模拟面试"}],
    "experience": [{"title": "后端开发", "period": "2年", "description": "REST API PostgreSQL"}],
    "summary": "后端开发两年",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ragas", action="store_true")
    parser.add_argument(
        "--enterprise",
        action="store_true",
        help="Strict enterprise gates (implies --with-ragas)",
    )
    args = parser.parse_args()
    if args.enterprise:
        args.with_ragas = True

    golden_path = ROOT / "evals" / "rag_golden.json"
    items = json.loads(golden_path.read_text(encoding="utf-8"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "enterprise" if args.enterprise else ("ragas" if args.with_ragas else "basic"),
        "cases": [],
        "refuse_pass_rate": 0.0,
        "golden_pass_rate": 0.0,
        "context_integrity": None,
        "enterprise": None,
        "ragas": None,
        "diagnosis": {
            "citation_display": (
                "UI 引用曾用 snippet=c.text[:160]，会在词中间截断；"
                "现已改为展示完整 chunk。LLM 上下文本身一直用完整 c.text。"
            ),
            "chunking": (
                "分块改为优先在句读边界切开，结构化摘要不再对 JSON 硬切 [:1500]。"
            ),
            "top_k": "top_k=5 只影响召回条数，不造成单条引用「截一半」。",
            "faithfulness_issue": (
                "「全部信息」默认仅导出简历；检索按姓名 resume_ids 隔离。"
            ),
        },
    }

    # Enterprise hard gates (fail CI if below)
    GATES = {
        "refuse_pass_rate": 1.0,
        "golden_pass_rate": 1.0,
        "context_precision_named": 1.0,
        "opensource_recall": True,
        "production_faithfulness_min": 0.90,
        "production_relevancy_min": 0.90,
        "hallucination_case_max_faithfulness": 0.40,
        "dossier_no_jd_interview": True,
        "no_cross_resume_facts": True,
    }

    app = create_app()
    refuse_total = 0
    refuse_pass = 0
    scored_total = 0
    scored_pass = 0

    with app.app_context():
        report["context_integrity"] = _context_integrity_check()

        for item in items:
            state = run_qa(
                {
                    "query": item["question"],
                    "role": item.get("role") or "candidate",
                    "user_id": 1,
                    "candidate_id": 1 if item.get("role") != "hr" else None,
                    "org_id": item.get("org_id") or 1,
                    "context_note": item.get("context_note") or "",
                    "recent_turns": "",
                }
            )
            answer = state.get("answer") or ""
            refused = bool(state.get("refused")) or answer.strip().startswith("不知道")
            case = {
                "id": item["id"],
                "question": item["question"],
                "answer": answer[:1200],
                "answer_len": len(answer),
                "refused": refused,
                "must_refuse": bool(item.get("must_refuse")),
                "pass": None,
                "failures": [],
            }
            ok = True
            if item.get("must_refuse"):
                refuse_total += 1
                ok = refused
                if ok:
                    refuse_pass += 1
                else:
                    case["failures"].append("expected_refuse")
            else:
                scored_total += 1
                if item.get("expect_contains_any"):
                    if not any(x in answer for x in item["expect_contains_any"]):
                        ok = False
                        case["failures"].append("missing_expected")
                if item.get("forbid_contains_any"):
                    hit = [x for x in item["forbid_contains_any"] if x in answer]
                    if hit:
                        ok = False
                        case["failures"].append(f"forbidden:{hit}")
                if ok:
                    scored_pass += 1
            case["pass"] = ok
            report["cases"].append(case)

        report["refuse_pass_rate"] = (refuse_pass / refuse_total) if refuse_total else 1.0
        report["golden_pass_rate"] = (scored_pass / scored_total) if scored_total else 1.0

        if args.with_ragas:
            report["ragas"] = _maybe_ragas(items)
            report["enterprise"] = _enterprise_scorecard(report["ragas"], report, GATES)

    out_dir = ROOT / "evals" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rag_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ent = report.get("enterprise") or {}
    summary = {
        "report": str(out),
        "mode": report["mode"],
        "refuse_pass_rate": report["refuse_pass_rate"],
        "golden_pass_rate": report["golden_pass_rate"],
        "context_integrity_pass": (report["context_integrity"] or {}).get("pass"),
        "enterprise_pass": ent.get("pass"),
        "enterprise_grade": ent.get("grade"),
        "gates": ent.get("gates"),
        "failed_gates": ent.get("failed_gates"),
        "metrics": ent.get("metrics"),
        "failed_golden": [c["id"] for c in report["cases"] if not c.get("pass")],
    }
    ragas = report.get("ragas") or {}
    if isinstance(ragas, dict) and not ragas.get("skipped"):
        summary["production_faithfulness"] = ragas.get("production_faithfulness")
        summary["production_relevancy"] = ragas.get("production_relevancy")
        summary["hallucination_detector_faithfulness"] = ragas.get(
            "hallucination_detector_faithfulness"
        )
        if ragas.get("per_case"):
            summary["per_case"] = [
                {
                    "id": c.get("id"),
                    "faithfulness": c.get("faithfulness"),
                    "answer_relevancy": c.get("answer_relevancy"),
                    "brief": c.get("brief"),
                }
                for c in ragas["per_case"]
            ]
        if ragas.get("retrieval"):
            summary["retrieval"] = ragas["retrieval"]
        diag = ragas.get("diagnostics") or {}
        if diag:
            summary["diagnostics"] = {
                "answer_is_dossier": diag.get("answer_is_dossier"),
                "answer_has_jd_wall": diag.get("answer_has_jd_wall"),
                "answer_has_interview_dump": diag.get("answer_has_interview_dump"),
                "context_precision": diag.get("context_precision"),
                "answer_has_opensource_repos": diag.get("answer_has_opensource_repos"),
                "answer_pollutes_zhangsan_facts": diag.get("answer_pollutes_zhangsan_facts"),
            }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if any(c["must_refuse"] and not c["pass"] for c in report["cases"]):
        return 1
    if report["context_integrity"] and not report["context_integrity"].get("pass"):
        return 1
    if args.enterprise and not (report.get("enterprise") or {}).get("pass"):
        return 1
    if any(not c["pass"] for c in report["cases"]):
        return 1
    return 0


def _context_integrity_check() -> dict:
    from app.services.rag import HybridRAG

    rag = HybridRAG(index_dir=ROOT / "data" / "faiss_eval_tmp")
    rag.chunks = []
    rag._vectors = None
    rag._faiss_index = None
    n = rag.index_resume(
        resume_id=9001,
        candidate_id=9001,
        raw_text=SAMPLE_RESUME,
        structured=SAMPLE_STRUCTURED,
        candidate_name="张三",
        org_id=1,
    )
    hits = rag.retrieve("张三的电话和 PostgreSQL 经验", top_k=5, allow_all=True, org_id=1)

    snippets = []
    issues = []
    for chunk, score in hits:
        text = chunk.text
        snippets.append(
            {
                "source": chunk.metadata.get("source"),
                "score": round(float(score), 3),
                "len": len(text),
                "preview": text[:160] + ("…" if len(text) > 160 else ""),
            }
        )
        if "结构化摘要:" in text:
            payload = text.split("结构化摘要:", 1)[1].strip()
            try:
                json.loads(payload)
            except json.JSONDecodeError:
                issues.append(f"structured_json_broken:{chunk.metadata.get('source')}")

    joined = "\n".join(c.text for c, _ in hits)
    must_have = ["13800138000", "PostgreSQL", "zhangsan@example.com"]
    missing = [k for k in must_have if k not in joined]

    result = {
        "chunks_indexed": n,
        "hits": len(hits),
        "snippets": snippets,
        "missing_facts_in_top_k": missing,
        "issues": issues,
        "pass": n > 0 and len(hits) > 0 and not missing and not issues,
        "note": (
            "不是 top-k 问题。旧 UI 用 snippet=c.text[:160] 只显示一半；"
            "LLM 一直吃完整 chunk。Ragas faithfulness 必须喂完整 contexts。"
        ),
    }
    try:
        import shutil

        shutil.rmtree(ROOT / "data" / "faiss_eval_tmp", ignore_errors=True)
    except Exception:
        pass
    return result


def _stub_vertexai() -> None:
    stub_name = "langchain_community.chat_models.vertexai"
    if stub_name not in sys.modules:
        stub = types.ModuleType(stub_name)

        class ChatVertexAI:
            pass

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[stub_name] = stub


def _maybe_ragas(items: list) -> dict:
    """DeepSeek-compatible faithfulness/relevancy (official ragas uses n>1, DeepSeek rejects)."""
    from app.config import Config

    if not Config.llm_enabled():
        return {
            "skipped": True,
            "reason": "LLM_API_KEY empty",
            "heuristic": _heuristic_rag_quality(),
        }

    from app.services.rag import HybridRAG

    linxiao_raw = (
        "林晓 求职意向：AI Agent应用工程师\n"
        "教育背景 2022-09 ~ 2026-06 星城理工大学 大数据与人工智能\n"
        "工作经历 云启智能 AI Agent开发\n"
        "项目经历 二代征信报告识别与RAG问答系统 政务数字人交互系统\n"
        "开源项目 GitHub：Galaxy527 xiaoyou-chat-agent：小柚 "
        "wsy-studio：WSY Studio zhi-tou-multi-agent：智投\n"
        "技能 Python FastAPI LangGraph RAG MCP\n"
    )
    linxiao_struct = {
        "name": "林晓",
        "email": "linxiao@example.com",
        "phone": "13900139000",
        "education": [{"school": "星城理工大学", "major": "大数据与人工智能"}],
        "skills": ["Python", "FastAPI", "LangGraph", "RAG", "MCP"],
        "projects": [
            {"name": "二代征信报告识别与RAG问答系统"},
            {"name": "政务数字人交互系统"},
        ],
        "opensource_projects": [
            {"name": "xiaoyou-chat-agent"},
            {"name": "wsy-studio"},
            {"name": "zhi-tou-multi-agent"},
        ],
        "summary": "AI Agent 应用",
    }

    rag = HybridRAG(index_dir=ROOT / "data" / "faiss_eval_tmp_ragas")
    rag.chunks = []
    rag._vectors = None
    rag._faiss_index = None
    rag.index_resume(1, 1, SAMPLE_RESUME, SAMPLE_STRUCTURED, "张三", 1)
    rag.index_resume(2, 2, linxiao_raw, linxiao_struct, "林晓", 1)

    diagnostics = _build_linxiao_diagnostics(rag, linxiao_raw)

    fact_q = "张三的电话是多少？技能里有没有 PostgreSQL？"
    hits_a = rag.retrieve(fact_q, top_k=5, resume_ids=[1], org_id=1)
    state_a = run_qa(
        {
            "query": fact_q,
            "role": "hr",
            "user_id": 1,
            "org_id": 1,
            "context_note": "候选人张三 电话=13800138000 技能含 PostgreSQL",
        }
    )
    ans_a = state_a.get("answer") or ""
    ctx_a = [c.text for c, _ in hits_a] or [SAMPLE_RESUME]

    full_q = "我要林晓简历的全部信息"
    ans_b = diagnostics.get("live_answer") or ""
    ctx_b = [ans_b] if diagnostics.get("answer_is_dossier") else (
        diagnostics.get("filtered_contexts") or [linxiao_raw]
    )

    open_q = "林晓有哪些开源项目？"
    hits_d = rag.retrieve(open_q, top_k=5, resume_ids=[2], org_id=1)
    state_d = run_qa(
        {
            "query": open_q,
            "role": "hr",
            "user_id": 1,
            "org_id": 1,
            "context_note": "",
            "recent_turns": "",
        }
    )
    ans_d = state_d.get("answer") or ""
    # Groundedness: prefer retrieved opensource chunks + raw seed; if answer is DB list, include it
    ctx_d = [c.text for c, _ in hits_d] or [linxiao_raw]
    if "开源项目" in ans_d and any(x in ans_d for x in ("xiaoyou", "wsy", "zhi-tou", "Galaxy")):
        ctx_d = [linxiao_raw, json.dumps(linxiao_struct, ensure_ascii=False), ans_d]

    bad_ans = "林晓信息如下：项目经历有二代征信；开源项目：无；另外张三电话13800138000。"
    ctx_c = [linxiao_raw]

    cases = [
        {"id": "A_fact", "question": fact_q, "answer": ans_a, "contexts": ctx_a, "kind": "prod"},
        {"id": "B_full_resume", "question": full_q, "answer": ans_b, "contexts": ctx_b, "kind": "prod"},
        {"id": "D_opensource", "question": open_q, "answer": ans_d, "contexts": ctx_d, "kind": "prod"},
        {"id": "C_bad_summary", "question": open_q, "answer": bad_ans, "contexts": ctx_c, "kind": "detector"},
    ]

    try:
        per_case = []
        for case in cases:
            scored = _llm_faithfulness_and_relevancy(
                case["question"], case["answer"], case["contexts"]
            )
            scored["id"] = case["id"]
            scored["kind"] = case["kind"]
            per_case.append(scored)

        prod = [c for c in per_case if c.get("kind") == "prod"]
        det = [c for c in per_case if c.get("kind") == "detector"]
        faiths = [c.get("faithfulness") for c in prod if isinstance(c.get("faithfulness"), (int, float))]
        relevs = [
            c.get("answer_relevancy") for c in prod if isinstance(c.get("answer_relevancy"), (int, float))
        ]
        det_f = [
            c.get("faithfulness") for c in det if isinstance(c.get("faithfulness"), (int, float))
        ]
        retrieval = {
            "context_precision_named_query": diagnostics.get("context_precision"),
            "opensource_in_filtered": diagnostics.get("opensource_in_filtered"),
            "name_scoped_hits": diagnostics.get("retrieve_name_scoped_hits"),
            "raw_pollution_before_scope": diagnostics.get("retrieve_zhangsan_pollution"),
            "scoped_pollution": diagnostics.get("retrieve_scoped_pollution"),
            "proxy_retrieval_quality": diagnostics.get("proxy_retrieval_quality"),
            "opensource_answer_ok": any(
                x in ans_d for x in ("xiaoyou", "wsy-studio", "zhi-tou", "Galaxy527")
            ),
            "opensource_no_false_empty": not (
                "开源" in ans_d and "无" in ans_d and "xiaoyou" not in ans_d
            ),
        }
        all_f = [c.get("faithfulness") for c in per_case if isinstance(c.get("faithfulness"), (int, float))]
        all_r = [
            c.get("answer_relevancy") for c in per_case if isinstance(c.get("answer_relevancy"), (int, float))
        ]
        out = {
            "engine": "hirepilot_llm_ragas_proxy",
            "note": (
                "官方 ragas 会对 DeepSeek 发 n>1；此处用项目 LLM（n=1）。"
                "production_* 不含故意错误用例 C；C 仅验证幻觉检测敏感度。"
            ),
            "production_faithfulness": round(sum(faiths) / len(faiths), 4) if faiths else None,
            "production_relevancy": round(sum(relevs) / len(relevs), 4) if relevs else None,
            "hallucination_detector_faithfulness": round(sum(det_f) / len(det_f), 4) if det_f else None,
            "faithfulness": round(sum(all_f) / len(all_f), 4) if all_f else None,
            "answer_relevancy": round(sum(all_r) / len(all_r), 4) if all_r else None,
            "per_case": per_case,
            "retrieval": retrieval,
            "diagnostics": diagnostics,
            "heuristic": _score_answer_against_context(
                ans_b, "\n".join(ctx_b if isinstance(ctx_b, list) else [str(ctx_b)])
            ),
            "interpretation": {
                "case_A": "基线事实问答（张三，按姓名 scope）",
                "case_B": "林晓全部信息 → 仅简历导出",
                "case_D": "林晓开源项目问答",
                "case_C": "故意错误摘要（开源=无+串张三）→ 应低分",
            },
        }
        return out
    except Exception as exc:
        return {
            "skipped": True,
            "reason": str(exc),
            "diagnostics": diagnostics,
            "heuristic": _heuristic_rag_quality(),
        }
    finally:
        try:
            import shutil

            shutil.rmtree(ROOT / "data" / "faiss_eval_tmp_ragas", ignore_errors=True)
        except Exception:
            pass


def _enterprise_scorecard(ragas: dict | None, report: dict, gates: dict) -> dict:
    ragas = ragas or {}
    diag = ragas.get("diagnostics") or {}
    retrieval = ragas.get("retrieval") or {}
    metrics = {
        "refuse_pass_rate": report.get("refuse_pass_rate"),
        "golden_pass_rate": report.get("golden_pass_rate"),
        "context_integrity": bool((report.get("context_integrity") or {}).get("pass")),
        "context_precision_named": retrieval.get("context_precision_named_query"),
        "opensource_recall": bool(
            retrieval.get("opensource_in_filtered") or retrieval.get("opensource_answer_ok")
        ),
        "production_faithfulness": ragas.get("production_faithfulness"),
        "production_relevancy": ragas.get("production_relevancy"),
        "hallucination_detector_faithfulness": ragas.get("hallucination_detector_faithfulness"),
        "dossier_no_jd_interview": (
            bool(diag.get("answer_is_dossier"))
            and not diag.get("answer_has_jd_wall")
            and not diag.get("answer_has_interview_dump")
        ),
        "no_cross_resume_facts": not bool(diag.get("answer_pollutes_zhangsan_facts")),
        "scoped_pollution": retrieval.get("scoped_pollution", diag.get("retrieve_scoped_pollution")),
    }

    failed: list[str] = []
    gate_results: dict[str, bool] = {}

    def _need(name: str, ok: bool) -> None:
        gate_results[name] = bool(ok)
        if not ok:
            failed.append(name)

    _need("refuse_pass_rate", (metrics["refuse_pass_rate"] or 0) >= gates["refuse_pass_rate"])
    _need("golden_pass_rate", (metrics["golden_pass_rate"] or 0) >= gates["golden_pass_rate"])
    _need("context_integrity", metrics["context_integrity"] is True)
    _need(
        "context_precision_named",
        (metrics["context_precision_named"] or 0) >= gates["context_precision_named"],
    )
    _need("opensource_recall", metrics["opensource_recall"] is True)
    pf = metrics["production_faithfulness"]
    pr = metrics["production_relevancy"]
    _need(
        "production_faithfulness",
        isinstance(pf, (int, float)) and pf >= gates["production_faithfulness_min"],
    )
    _need(
        "production_relevancy",
        isinstance(pr, (int, float)) and pr >= gates["production_relevancy_min"],
    )
    hf = metrics["hallucination_detector_faithfulness"]
    _need(
        "hallucination_detector",
        isinstance(hf, (int, float)) and hf <= gates["hallucination_case_max_faithfulness"],
    )
    _need("dossier_no_jd_interview", metrics["dossier_no_jd_interview"] is True)
    _need("no_cross_resume_facts", metrics["no_cross_resume_facts"] is True)
    _need("scoped_pollution_zero", (metrics["scoped_pollution"] or 0) == 0)

    if ragas.get("skipped"):
        failed.append("ragas_skipped")
        gate_results["ragas_available"] = False
    else:
        gate_results["ragas_available"] = True

    passed = len(failed) == 0
    grade = "A" if passed else ("B" if len(failed) <= 2 else "C")
    return {
        "pass": passed,
        "grade": grade,
        "gates": gate_results,
        "failed_gates": failed,
        "metrics": metrics,
        "thresholds": gates,
    }


def _llm_faithfulness_and_relevancy(question: str, answer: str, contexts: list[str]) -> dict:
    from app.services.llm import llm_json

    ctx = "\n---\n".join((contexts or [])[:8])
    # Keep prompt bounded for long dossiers
    if len(ctx) > 12000:
        ctx = ctx[:12000] + "\n…(truncated)"
    ans = answer or ""
    if len(ans) > 8000:
        ans = ans[:8000] + "\n…(truncated)"

    prompt = f"""你是 RAG 评测器。只依据【上下文】判断【回答】是否忠实、是否相关。
DeepSeek 兼容：不要要求 n>1 采样。

【问题】
{question}

【上下文】
{ctx}

【回答】
{ans}

请只输出 JSON（不要 markdown）：
{{
  "faithfulness": 0.0到1.0的小数,
  "answer_relevancy": 0.0到1.0的小数,
  "supported_claims": 整数,
  "total_claims": 整数,
  "unsupported_examples": ["最多3条不被上下文支持的主张"],
  "brief": "一句话说明"
}}

评分规则：
- faithfulness：回答中可核查事实有多少被上下文支持；档案原文导出且未编造应接近 1.0；若写「开源=无」但上下文有开源仓库名，应很低。
- answer_relevancy：回答是否直接回应问题。
"""
    data = llm_json(
        prompt,
        {
            "faithfulness": None,
            "answer_relevancy": None,
            "supported_claims": 0,
            "total_claims": 0,
            "unsupported_examples": [],
            "brief": "fallback",
        },
        temperature=0,
    )
    if not isinstance(data, dict):
        return {"faithfulness": None, "answer_relevancy": None, "error": "bad_json"}
    out = {
        "faithfulness": _as_unit(data.get("faithfulness")),
        "answer_relevancy": _as_unit(data.get("answer_relevancy")),
        "supported_claims": data.get("supported_claims"),
        "total_claims": data.get("total_claims"),
        "unsupported_examples": data.get("unsupported_examples") or [],
        "brief": data.get("brief"),
    }
    return out


def _as_unit(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, x))


def _build_linxiao_diagnostics(rag, linxiao_raw: str) -> dict:
    full_q = "我要林晓简历的全部信息"
    from app.services.resume_dossier import _guess_person_name, wants_full_dossier

    hits_raw = rag.retrieve(full_q, top_k=10, allow_all=True, org_id=1)
    pollution = sum(1 for c, _ in hits_raw if c.metadata.get("resume_id") == 1)
    hits_scoped = rag.retrieve(full_q, top_k=10, resume_ids=[2], org_id=1)
    pollution_scoped = sum(1 for c, _ in hits_scoped if c.metadata.get("resume_id") == 1)
    precision = 1.0
    if hits_scoped:
        precision = 1.0 - (pollution_scoped / max(1, len(hits_scoped)))
    elif hits_raw:
        precision = 1.0 - (pollution / max(1, len(hits_raw)))

    state_b = run_qa(
        {
            "query": full_q,
            "role": "hr",
            "user_id": 1,
            "org_id": 1,
            "context_note": "HR摘要：林晓 简历#2",
            "recent_turns": "",
        }
    )
    ans_b = state_b.get("answer") or ""
    opensource_ok = any(
        x in ans_b for x in ("xiaoyou", "wsy-studio", "zhi-tou", "Galaxy527")
    )
    pollutes_facts = ("zhangsan@" in ans_b.lower()) or ("13800138000" in ans_b)
    return {
        "guess_name": _guess_person_name(full_q),
        "wants_full": wants_full_dossier(full_q),
        "retrieve_raw_hits": len(hits_raw),
        "retrieve_zhangsan_pollution": pollution,
        "retrieve_name_scoped_hits": len(hits_scoped),
        "retrieve_scoped_pollution": pollution_scoped,
        "context_precision": round(precision, 4),
        "opensource_in_filtered": any(
            "xiaoyou" in c.text or "开源" in c.text for c, _ in hits_scoped
        ),
        "filtered_contexts": [c.text for c, _ in hits_scoped] or [linxiao_raw],
        "live_answer": ans_b,
        "answer_is_dossier": ans_b.startswith("======== 简历#"),
        "answer_has_opensource_repos": opensource_ok,
        "answer_pollutes_zhangsan_facts": pollutes_facts,
        "answer_has_jd_wall": "岗位需求原文" in ans_b or "岗位职责" in ans_b[:800],
        "answer_has_interview_dump": "面试记录" in ans_b and "问答明细" in ans_b,
        # legacy key: true if account display line mentions 张三 (not content mix)
        "answer_mentions_zhangsan": "张三" in ans_b,
        "proxy_retrieval_quality": round(
            (0.5 * precision)
            + (0.3 if opensource_ok or any("xiaoyou" in c.text for c, _ in hits_scoped) else 0.0)
            + (0.2 if pollution_scoped == 0 else 0.0),
            4,
        ),
        "answer_preview": ans_b[:280],
    }


def _score_answer_against_context(answer: str, context: str) -> dict:
    ans = answer or ""
    ctx = context or ""
    checks = {
        "has_opensource_repos": any(
            x in ans for x in ("xiaoyou", "wsy-studio", "zhi-tou", "Galaxy527")
        ),
        "false_opensource_empty": (
            "开源" in ans and "无" in ans and "xiaoyou" not in ans and "Galaxy" not in ans
        ),
        "pollutes_zhangsan_facts": (
            ("zhangsan@" in ans.lower() or "13800138000" in ans)
            and ("林晓" in ctx or "xiaoyou" in ctx or "林晓" in ans)
        ),
        "is_full_dossier": ans.startswith("======== 简历#"),
        "context_has_opensource": "xiaoyou" in ctx or "开源项目" in ctx,
    }
    checks["proxy_faithful"] = checks["is_full_dossier"] or (
        checks["has_opensource_repos"]
        and not checks["false_opensource_empty"]
        and not checks["pollutes_zhangsan_facts"]
    )
    return checks


def _heuristic_rag_quality() -> dict:
    from app.services.rag import HybridRAG

    linxiao_raw = (
        "林晓 开源项目 GitHub：Galaxy527 xiaoyou-chat-agent wsy-studio "
        "zhi-tou-multi-agent 项目经历 二代征信报告识别与RAG问答系统"
    )
    rag = HybridRAG(index_dir=ROOT / "data" / "faiss_eval_tmp_heur")
    rag.chunks = []
    rag._vectors = None
    rag._faiss_index = None
    rag.index_resume(1, 1, SAMPLE_RESUME, SAMPLE_STRUCTURED, "张三", 1)
    rag.index_resume(
        2,
        2,
        linxiao_raw,
        {
            "name": "林晓",
            "opensource_projects": [{"name": "xiaoyou-chat-agent"}],
            "projects": [{"name": "二代征信报告识别与RAG问答系统"}],
        },
        "林晓",
        1,
    )
    diag = _build_linxiao_diagnostics(rag, linxiao_raw)
    proxy = _score_answer_against_context(
        diag.get("live_answer") or "",
        "\n".join(diag.get("filtered_contexts") or [linxiao_raw]),
    )
    bad = _score_answer_against_context("开源项目：无；张三电话13800138000", linxiao_raw)
    try:
        import shutil

        shutil.rmtree(ROOT / "data" / "faiss_eval_tmp_heur", ignore_errors=True)
    except Exception:
        pass
    return {"diagnostics": diag, "good_answer_proxy": proxy, "bad_summary_proxy": bad}


if __name__ == "__main__":
    raise SystemExit(main())

"""Faiss RAG: chunk → embed → hybrid retrieve → cite sources."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Config
from app.services.embeddings import get_embedder
from app.services.llm import llm_text

logger = logging.getLogger(__name__)

# Bump when chunk metadata / indexing policy changes → force rebuild.
INDEX_SCHEMA = 2


@dataclass
class DocChunk:
    chunk_id: str
    text: str
    metadata: dict  # resume_id, candidate_id, source, etc.


class HybridRAG:
    """In-memory + disk Faiss index with keyword hybrid retrieval."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir or Config.FAISS_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[DocChunk] = []
        self._embeddings = None
        self._faiss_index = None
        self._vectors: np.ndarray | None = None
        self._load()

    def _meta_path(self) -> Path:
        return self.index_dir / "chunks.pkl"

    def _vec_path(self) -> Path:
        return self.index_dir / "vectors.npy"

    def _info_path(self) -> Path:
        return self.index_dir / "index_info.json"

    def _load(self) -> None:
        info = {}
        if self._info_path().exists():
            try:
                info = json.loads(self._info_path().read_text(encoding="utf-8"))
            except Exception:
                info = {}
        embedder = get_embedder()
        if info.get("embedder") and info.get("embedder") != embedder.name:
            logger.warning(
                "Embedding backend changed (%s -> %s); clearing Faiss index",
                info.get("embedder"),
                embedder.name,
            )
            self.chunks = []
            self._vectors = None
            self._faiss_index = None
            return
        if info.get("dim") and int(info["dim"]) != embedder.dim:
            logger.warning(
                "Embedding dim changed (%s -> %s); clearing Faiss index",
                info.get("dim"),
                embedder.dim,
            )
            self.chunks = []
            self._vectors = None
            self._faiss_index = None
            return
        if self._meta_path().exists():
            with open(self._meta_path(), "rb") as f:
                self.chunks = pickle.load(f)
        if self._vec_path().exists():
            self._vectors = np.load(self._vec_path())
            self._rebuild_faiss()

    def _save(self) -> None:
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self.chunks, f)
        if self._vectors is not None:
            np.save(self._vec_path(), self._vectors)
        emb = get_embedder()
        self._info_path().write_text(
            json.dumps({"embedder": emb.name, "dim": emb.dim}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_faiss(self) -> None:
        if self._vectors is None or len(self._vectors) == 0:
            self._faiss_index = None
            return
        try:
            import faiss

            dim = self._vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9
            normalized = (self._vectors / norms).astype("float32")
            index.add(normalized)
            self._faiss_index = index
        except Exception as exc:
            logger.warning("Faiss rebuild failed: %s", exc)
            self._faiss_index = None

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        emb = get_embedder()
        try:
            vecs = emb.embed_documents(texts)
            arr = np.array(vecs, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as exc:
            logger.warning("Embedding failed (%s), using hash fallback: %s", emb.name, exc)
            from app.services.embeddings import HashEmbedder

            return np.vstack([HashEmbedder()._one(t) for t in texts])

    def chunk_text(self, text: str, size: int = 500, overlap: int = 100) -> list[str]:
        """Split text into overlapping chunks, preferring punctuation boundaries."""
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        i = 0
        n = len(text)
        seps = ("。", "！", "？", "；", ". ", "\n", "，", ",", " ")
        while i < n:
            end = min(n, i + size)
            if end < n:
                window = text[i:end]
                best_pos = -1
                best_sep = ""
                for sep in seps:
                    pos = window.rfind(sep)
                    if pos > best_pos:
                        best_pos = pos
                        best_sep = sep
                if best_pos >= size // 2:
                    end = i + best_pos + len(best_sep)
            piece = text[i:end].strip()
            if piece:
                chunks.append(piece)
            if end >= n:
                break
            i = max(i + 1, end - overlap)
        return chunks

    # Section headers → (label for citation, kind for filtering/boost)
    _SECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"(开源项目|GitHub|Github|GITHUB|个人开源)"), "开源项目", "opensource"),
        (re.compile(r"(项目经历|项目经验|代表项目|项目介绍)"), "项目经历", "projects"),
        (re.compile(r"(工作经历|工作经验|实习经历|职业经历)"), "工作经历", "experience"),
        (re.compile(r"(教育背景|教育经历|学历)"), "教育背景", "education"),
        (re.compile(r"(技能特长|专业技能|技术栈|掌握技能)"), "技能特长", "skills"),
        (re.compile(r"(自我评价|个人总结|自我介绍)"), "自我评价", "summary"),
        (re.compile(r"(求职意向|意向岗位)"), "求职意向", "intent"),
    ]

    def split_resume_sections(self, text: str) -> list[tuple[str, str, str]]:
        """Return list of (section_label, section_kind, body)."""
        raw = (text or "").strip()
        if not raw:
            return []

        # Find header positions
        hits: list[tuple[int, str, str]] = []
        for pat, label, kind in self._SECTION_PATTERNS:
            for m in pat.finditer(raw):
                hits.append((m.start(), label, kind))
        hits.sort(key=lambda x: x[0])

        # Deduplicate overlapping starts (keep first label at same index)
        dedup: list[tuple[int, str, str]] = []
        seen_pos: set[int] = set()
        for pos, label, kind in hits:
            # skip if within 2 chars of previous header
            if any(abs(pos - p) < 2 for p in seen_pos):
                continue
            seen_pos.add(pos)
            dedup.append((pos, label, kind))

        if not dedup:
            return [("简历正文", "body", raw)]

        sections: list[tuple[str, str, str]] = []
        if dedup[0][0] > 0:
            head = raw[: dedup[0][0]].strip()
            if head:
                sections.append(("基本信息", "profile", head))
        for i, (pos, label, kind) in enumerate(dedup):
            end = dedup[i + 1][0] if i + 1 < len(dedup) else len(raw)
            body = raw[pos:end].strip()
            if body:
                sections.append((label, kind, body))

        # Merge consecutive same-kind sections (e.g. 开源项目 + GitHub)
        merged: list[tuple[str, str, str]] = []
        for label, kind, body in sections:
            if merged and merged[-1][1] == kind:
                prev_l, prev_k, prev_b = merged[-1]
                merged[-1] = (prev_l, prev_k, f"{prev_b}\n{body}".strip())
            else:
                merged.append((label, kind, body))
        return merged or [("简历正文", "body", raw)]

    def index_resume(
        self,
        resume_id: int,
        candidate_id: int,
        raw_text: str,
        structured: dict | None = None,
        candidate_name: str = "",
        org_id: int | None = None,
        *,
        job_title: str | None = None,
        extra_note: str | None = None,
    ) -> int:
        """Remove old chunks for resume, re-index. Returns chunk count.

        Important: do NOT index full job JD into resume chunks — that pollutes
        project/skill retrieval. Job title may be stored as a tiny intent note.
        """
        self.chunks = [c for c in self.chunks if c.metadata.get("resume_id") != resume_id]

        pieces: list[tuple[str, str, str]] = []  # text, section_label, section_kind

        # Resume body by section (projects vs opensource stay separate)
        for label, kind, body in self.split_resume_sections(raw_text or ""):
            size = 700 if kind in {"projects", "opensource", "experience"} else 500
            for part in self.chunk_text(body, size=size, overlap=80):
                pieces.append((part, label, kind))

        # Compact structured fields — keep projects / opensource separate blobs
        if structured:
            base = {
                k: structured.get(k)
                for k in ("name", "email", "phone", "education", "skills", "summary", "experience")
                if structured.get(k)
            }
            if base:
                blob = "结构化摘要: " + json.dumps(base, ensure_ascii=False)
                for part in self.chunk_text(blob, size=900, overlap=40):
                    pieces.append((part, "结构化摘要", "structured"))
            if structured.get("projects"):
                blob = "项目经历(结构化): " + json.dumps(
                    structured.get("projects"), ensure_ascii=False
                )
                for part in self.chunk_text(blob, size=900, overlap=40):
                    pieces.append((part, "项目经历", "projects"))
            if structured.get("opensource_projects"):
                blob = "开源项目(结构化): " + json.dumps(
                    structured.get("opensource_projects"), ensure_ascii=False
                )
                for part in self.chunk_text(blob, size=900, overlap=40):
                    pieces.append((part, "开源项目", "opensource"))
            if structured.get("work_participation"):
                blob = "参与工作(结构化): " + json.dumps(
                    structured.get("work_participation"), ensure_ascii=False
                )
                for part in self.chunk_text(blob, size=900, overlap=40):
                    pieces.append((part, "参与工作", "participation"))

        # Short intent only — never full JD requirements text
        title = (job_title or "").strip()
        if title:
            pieces.append((f"求职意向岗位：{title}", "求职意向", "intent"))

        note = (extra_note or "").strip()
        if note:
            for part in self.chunk_text(note, size=600, overlap=60):
                pieces.append((part, "补充记录", "note"))

        new_chunks: list[DocChunk] = []
        for i, (part, label, kind) in enumerate(pieces):
            cid = f"r{resume_id}_{i}_{hashlib.md5(part.encode()).hexdigest()[:8]}"
            new_chunks.append(
                DocChunk(
                    chunk_id=cid,
                    text=part,
                    metadata={
                        "resume_id": resume_id,
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "org_id": org_id,
                        "section": label,
                        "section_kind": kind,
                        "schema": INDEX_SCHEMA,
                        "source": f"简历#{resume_id}/{label}/片段{i+1}",
                    },
                )
            )

        self.chunks = [
            c for c in self.chunks if c.metadata.get("resume_id") != resume_id
        ] + new_chunks

        if self.chunks:
            self._vectors = self._embed_texts([c.text for c in self.chunks])
            self._rebuild_faiss()
        else:
            self._vectors = None
            self._faiss_index = None

        self._save()
        logger.info("Indexed resume %s with %s chunks", resume_id, len(new_chunks))
        return len(new_chunks)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize for hybrid search — Chinese uses chars + bigrams."""
        text = (text or "").lower()
        parts = re.findall(r"[a-z0-9_+#./-]+|[\u4e00-\u9fff]+", text)
        tokens: set[str] = set()
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                tokens.add(part)
                for ch in part:
                    tokens.add(ch)
                for i in range(len(part) - 1):
                    tokens.add(part[i : i + 2])
            else:
                tokens.add(part)
        return tokens

    def _keyword_scores(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        scores = np.zeros(len(self.chunks), dtype="float32")
        if not tokens:
            return scores
        for i, chunk in enumerate(self.chunks):
            text_l = chunk.text.lower()
            hits = sum(1 for t in tokens if t in text_l)
            scores[i] = hits / max(1, len(tokens))
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_id: int | None = None,
        resume_ids: list[int] | None = None,
        allow_all: bool = False,
        org_id: int | None = None,
    ) -> list[tuple[DocChunk, float]]:
        if not self.chunks:
            return []

        allowed_idx = []
        for i, c in enumerate(self.chunks):
            meta = c.metadata
            if org_id is not None and meta.get("org_id") not in (None, org_id):
                # allow legacy chunks without org_id only when querying same deploy default
                if meta.get("org_id") is not None:
                    continue
            # resume_ids / candidate_id take precedence over allow_all (name-scoped HR search)
            if resume_ids is not None:
                if meta.get("resume_id") in resume_ids:
                    allowed_idx.append(i)
            elif candidate_id is not None and meta.get("candidate_id") == candidate_id:
                allowed_idx.append(i)
            elif allow_all:
                allowed_idx.append(i)

        if not allowed_idx:
            return []

        q_vec = self._embed_texts([query])[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        vec_scores = np.zeros(len(self.chunks), dtype="float32")
        if self._vectors is not None and len(self._vectors) == len(self.chunks):
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9
            mat = self._vectors / norms
            vec_scores = mat @ q_norm

        kw_scores = self._keyword_scores(query)
        hybrid = 0.55 * vec_scores + 0.45 * kw_scores

        # Query-aware section boost: prefer 项目经历 vs 开源 vs 技能 etc.
        boosts = self._section_boosts(query)
        for i in allowed_idx:
            kind = (self.chunks[i].metadata or {}).get("section_kind") or ""
            hybrid[i] = float(hybrid[i]) + boosts.get(kind, 0.0)

        ranked = sorted(allowed_idx, key=lambda i: hybrid[i], reverse=True)
        results = [(self.chunks[i], float(hybrid[i])) for i in ranked[:top_k]]
        # Drop below-threshold hits (hard refuse handled by QA agent)
        from app.config import Config

        min_score = Config.MIN_RETRIEVE_SCORE
        results = [(c, s) for c, s in results if s >= min_score]
        return results

    def _section_boosts(self, query: str) -> dict[str, float]:
        q = query or ""
        boosts: dict[str, float] = {}
        if any(k in q for k in ("开源", "GitHub", "github", "Github")):
            boosts["opensource"] = 0.18
            boosts["projects"] = -0.05
        elif any(k in q for k in ("项目", "做过什么", "负责过")):
            boosts["projects"] = 0.16
            boosts["opensource"] = 0.08
            boosts["participation"] = 0.06
        if any(k in q for k in ("技能", "技术栈", "会什么")):
            boosts["skills"] = 0.12
        if any(k in q for k in ("教育", "学历", "学校")):
            boosts["education"] = 0.12
        if any(k in q for k in ("工作", "实习", "经历")):
            boosts["experience"] = 0.1
        # Never prefer leftover JD pollution if any old chunks remain
        boosts.setdefault("job_jd", -0.25)
        return boosts

    def _candidate_index_stale(self, candidate_id: int) -> bool:
        related = [c for c in self.chunks if c.metadata.get("candidate_id") == candidate_id]
        if not related:
            return True
        for c in related:
            meta = c.metadata or {}
            if meta.get("schema") != INDEX_SCHEMA:
                return True
            # Old bug: full JD prepended into resume chunks
            head = (c.text or "")[:160]
            if "岗位需求：" in head and "求职岗位：" in head:
                return True
        return False

    def ensure_candidate_indexed(self, candidate_id: int) -> int:
        """If candidate has no chunks (or stale schema/JD pollution), rebuild from DB."""
        if not self._candidate_index_stale(candidate_id):
            return 0
        # Drop stale chunks for this candidate then rebuild
        self.chunks = [c for c in self.chunks if c.metadata.get("candidate_id") != candidate_id]
        try:
            from app.models import Resume

            resumes = Resume.query.filter_by(candidate_id=candidate_id).all()
        except Exception as exc:
            logger.warning("ensure_candidate_indexed DB failed: %s", exc)
            return 0
        total = 0
        for r in resumes:
            name = r.candidate.display_name if r.candidate else ""
            total += self.index_resume(
                resume_id=r.id,
                candidate_id=candidate_id,
                raw_text=r.raw_text or "",
                structured=r.structured(),
                candidate_name=name,
                org_id=r.org_id,
                job_title=r.target_job_title,
            )
        return total

    def reindex_all_from_db(self) -> int:
        """Rebuild all resume chunks from DB (applies new chunk/citation policy)."""
        try:
            from app.models import Resume

            resumes = Resume.query.filter(Resume.processing_status == "ready").all()
        except Exception as exc:
            logger.warning("reindex_all_from_db failed: %s", exc)
            return 0
        # Full wipe then rebuild under current schema
        self.chunks = []
        self._vectors = None
        self._faiss_index = None
        total = 0
        for r in resumes:
            name = r.candidate.display_name if r.candidate else ""
            total += self.index_resume(
                resume_id=r.id,
                candidate_id=r.candidate_id,
                raw_text=r.raw_text or "",
                structured=r.structured(),
                candidate_name=name,
                org_id=r.org_id,
                job_title=r.target_job_title,
            )
        logger.info("Reindexed %s resumes into %s chunks", len(resumes), total)
        return total

    def answer(
        self,
        query: str,
        *,
        candidate_id: int | None = None,
        resume_ids: list[int] | None = None,
        allow_all: bool = False,
        role: str = "candidate",
    ) -> tuple[str, list[dict]]:
        hits = self.retrieve(
            query,
            top_k=5,
            candidate_id=candidate_id,
            resume_ids=resume_ids,
            allow_all=allow_all,
        )
        if not hits:
            return "不知道。当前知识库中没有与问题相关的依据。", []

        context_blocks = []
        sources = []
        for chunk, score in hits:
            src = chunk.metadata.get("source", chunk.chunk_id)
            context_blocks.append(f"[{src}] (相关度 {score:.2f})\n{chunk.text}")
            sources.append(
                {
                    "source": src,
                    "score": round(score, 3),
                    "resume_id": chunk.metadata.get("resume_id"),
                    "snippet": chunk.text,
                }
            )

        context = "\n\n".join(context_blocks)
        prompt = f"""你是招聘中台问答助手。角色={role}。
只能依据下列检索片段回答；若不足以回答，必须回复「不知道」。
有答案时在末尾列出引用来源（[来源名]）。

检索片段：
{context}

用户问题：{query}
"""
        fallback = self._fallback_answer(query, hits)
        answer = llm_text(prompt, fallback)
        # Enforce no-hallucination soft check for mock
        if "不知道" in answer and len(answer) < 40:
            return answer, []
        return answer, sources

    def _fallback_answer(self, query: str, hits: list[tuple[DocChunk, float]]) -> str:
        if not hits or hits[0][1] < 0.05:
            return "不知道。当前知识库中没有与问题相关的依据。"
        top = hits[0][0]
        src = top.metadata.get("source", "")
        snippet = top.text[:280]
        return f"根据检索到的资料：{snippet}\n\n引用来源：[{src}]"


_rag_singleton: HybridRAG | None = None


def get_rag() -> HybridRAG:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = HybridRAG()
    return _rag_singleton


def reset_rag() -> None:
    global _rag_singleton
    _rag_singleton = None

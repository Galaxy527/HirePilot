"""Embedding backends: local sentence-transformers, OpenAI-compatible API, or hash."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Protocol

import numpy as np

from app.config import Config

logger = logging.getLogger(__name__)

_local_lock = threading.Lock()
_local_model = None
_local_dim: int | None = None


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "hash-md5"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t).tolist() for t in texts]

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec


class LocalSentenceEmbedder:
    def __init__(self, model_name: str):
        global _local_model, _local_dim
        with _local_lock:
            if _local_model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model: %s", model_name)
                _local_model = SentenceTransformer(model_name)
                probe = _local_model.encode(["ping"], normalize_embeddings=True)
                _local_dim = int(probe.shape[1])
                logger.info("Local embedding ready dim=%s", _local_dim)
            self._model = _local_model
            self._dim = int(_local_dim or 384)
            self._name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"local:{self._name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts or [""],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.astype("float32").tolist() for v in np.atleast_2d(vecs)]


class ApiEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str):
        from langchain_openai import OpenAIEmbeddings

        self._emb = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
        self._model = model
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.embed_documents(["dim-probe"])[0]
            self._dim = len(v)
        return self._dim

    @property
    def name(self) -> str:
        return f"api:{self._model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Resolve embedding backend once per process."""
    global _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        backend = Config.EMBEDDING_BACKEND

        if backend == "hash":
            _embedder = HashEmbedder()
            logger.info("Using hash embeddings dim=%s", _embedder.dim)
            return _embedder

        if backend == "api" and Config.EMBEDDING_MODEL and Config.EMBEDDING_API_KEY:
            try:
                _embedder = ApiEmbedder(
                    Config.EMBEDDING_MODEL,
                    Config.EMBEDDING_API_KEY,
                    Config.EMBEDDING_BASE_URL,
                )
                _ = _embedder.dim
                logger.info("Using API embeddings: %s dim=%s", _embedder.name, _embedder.dim)
                return _embedder
            except Exception as exc:
                logger.warning("API embeddings failed, trying local: %s", exc)

        if backend in {"local", "api"}:
            try:
                _embedder = LocalSentenceEmbedder(Config.LOCAL_EMBEDDING_MODEL)
                logger.info("Using local embeddings: %s dim=%s", _embedder.name, _embedder.dim)
                return _embedder
            except Exception as exc:
                logger.warning("Local embeddings unavailable, falling back to hash: %s", exc)

        _embedder = HashEmbedder()
        logger.info("Using hash embeddings dim=%s", _embedder.dim)
        return _embedder


def reset_embedder() -> None:
    global _embedder
    with _embedder_lock:
        _embedder = None

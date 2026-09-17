"""Embedding backends for the RAG index."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix."""


class TfidfEmbedder:
    """Sparse TF-IDF projected to dense vectors. Deterministic and local."""

    name = "tfidf"

    def __init__(self, max_features: int = 4096) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=max_features,
            dtype=np.float32,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        corpus = texts or [""]
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32, copy=False)
        if dense.ndim == 1:
            dense = dense.reshape(1, -1)
        if dense.shape[0] == 0:
            return np.zeros((0, int(getattr(self._vectorizer, "max_features", 1))), dtype=np.float32)
        return dense


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers when the package is available."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Tiny deterministic embedder for tests (character n-gram hashing)."""

    name = "hashing"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = (text or "").lower()
            for index in range(len(lowered) - 2):
                gram = lowered[index : index + 3]
                bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.dim
                matrix[row, bucket] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


def default_embedder(model_name: str | None = None) -> Embedder:
    """Prefer sentence-transformers; otherwise use local TF-IDF."""

    name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    if name in {"tfidf", "hashing"}:
        return TfidfEmbedder() if name == "tfidf" else HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(name)
    except Exception:
        return TfidfEmbedder()

"""FAISS vector store with a NumPy cosine fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from seismic_damage.io_utils import ensure_parent

_FAISS_ERROR: str | None
try:
    import faiss  # type: ignore

    _FAISS_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    faiss = None
    _FAISS_ERROR = str(exc)


class FaissVectorStore:
    """Inner-product store over L2-normalized embeddings."""

    def __init__(self, dim: int, use_faiss: bool | None = None) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim
        self.ids: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._use_faiss = bool(faiss is not None) if use_faiss is None else bool(use_faiss and faiss)
        self._index = faiss.IndexFlatIP(dim) if self._use_faiss else None

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    @property
    def size(self) -> int:
        return len(self.ids)

    def add(self, vectors: np.ndarray, ids: list[str], payloads: list[dict[str, Any]]) -> int:
        if len(ids) != len(payloads):
            raise ValueError("ids and payloads must have the same length")
        array = _as_normalized(vectors, self.dim)
        if array.shape[0] != len(ids):
            raise ValueError("vector rows must match id count")
        if self._index is not None:
            self._index.add(array)
        self._matrix = np.vstack([self._matrix, array]) if self.size else array
        self.ids.extend(ids)
        self.payloads.extend(payloads)
        return array.shape[0]

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float, dict[str, Any]]]:
        if self.size == 0 or top_k <= 0:
            return []
        q = _as_normalized(query, self.dim)
        k = min(top_k, self.size)
        if self._index is not None:
            scores, indices = self._index.search(q, k)
            ranked = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            sims = self._matrix @ q[0]
            order = np.argsort(-sims)[:k]
            ranked = [(int(i), float(sims[i])) for i in order]
        return [(self.ids[i], score, self.payloads[i]) for i, score in ranked]

    def save(self, directory: Path | str) -> Path:
        path = ensure_parent(Path(directory) / "meta.json")
        directory = path.parent
        np.save(directory / "vectors.npy", self._matrix)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / "index.faiss"))
        meta = {
            "dim": self.dim,
            "backend": self.backend,
            "ids": self.ids,
            "payloads": self.payloads,
            "faiss_import_error": _FAISS_ERROR,
        }
        path.write_text(json.dumps(meta, default=str), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> FaissVectorStore:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]), use_faiss=meta.get("backend") == "faiss")
        store.ids = list(meta.get("ids") or [])
        store.payloads = list(meta.get("payloads") or [])
        vectors_path = directory / "vectors.npy"
        if vectors_path.exists():
            store._matrix = np.load(vectors_path).astype(np.float32, copy=False)
        faiss_path = directory / "index.faiss"
        if store._index is not None and faiss_path.exists():
            store._index = faiss.read_index(str(faiss_path))
        elif store.size and store._index is not None:
            store._index.add(store._matrix)
        return store


def _as_normalized(vectors: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != dim:
        raise ValueError(f"expected dim={dim}, got {array.shape[1]}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import AppError


@dataclass(slots=True)
class KnowledgeChunk:
    doc_id: str
    title: str
    source: str
    chunk_id: str
    text: str
    vector: list[float]


@dataclass(slots=True)
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float

    @property
    def snippet(self) -> str:
        text = self.chunk.text.strip()
        if len(text) <= 220:
            return text
        return f"{text[:217].rstrip()}..."


class KnowledgeBaseService:
    def __init__(self, *, index_path: str, score_threshold: float) -> None:
        self._index_path = Path(index_path)
        self._score_threshold = score_threshold
        self._chunks: list[KnowledgeChunk] = []
        self._vector_dim: int | None = None
        self._ready = False
        self._not_ready_reason = "Knowledge base index is not ready."
        self._load_index()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def not_ready_reason(self) -> str:
        return self._not_ready_reason

    def _set_not_ready(self, reason: str) -> None:
        self._ready = False
        self._chunks = []
        self._vector_dim = None
        self._not_ready_reason = reason

    def _load_index(self) -> None:
        if not self._index_path.exists():
            self._set_not_ready(
                f"Knowledge base index file was not found at `{self._index_path}`."
            )
            return

        try:
            raw_payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_not_ready(
                f"Knowledge base index could not be loaded: {exc}"
            )
            return

        if isinstance(raw_payload, dict):
            raw_chunks = raw_payload.get("chunks")
        else:
            raw_chunks = raw_payload

        if not isinstance(raw_chunks, list):
            self._set_not_ready("Knowledge base index has an invalid structure.")
            return

        if not raw_chunks:
            self._set_not_ready("Knowledge base index is empty.")
            return

        parsed_chunks: list[KnowledgeChunk] = []
        vector_dim: int | None = None

        for item in raw_chunks:
            if not isinstance(item, dict):
                self._set_not_ready("Knowledge base index contains invalid chunk items.")
                return

            try:
                vector = self._validate_vector(item.get("vector"))
                current_dim = len(vector)
                if vector_dim is None:
                    vector_dim = current_dim
                elif vector_dim != current_dim:
                    self._set_not_ready(
                        "Knowledge base index has inconsistent vector dimensions."
                    )
                    return

                parsed_chunks.append(
                    KnowledgeChunk(
                        doc_id=str(item["doc_id"]),
                        title=str(item["title"]),
                        source=str(item["source"]),
                        chunk_id=str(item["chunk_id"]),
                        text=str(item["text"]),
                        vector=vector,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._set_not_ready(
                    f"Knowledge base index contains invalid chunk data: {exc}"
                )
                return

        if vector_dim is None:
            self._set_not_ready("Knowledge base index does not contain valid vectors.")
            return

        self._chunks = parsed_chunks
        self._vector_dim = vector_dim
        self._ready = True
        self._not_ready_reason = ""

    @staticmethod
    def _validate_vector(raw_vector: Any) -> list[float]:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError("vector must be a non-empty list")
        vector = [float(value) for value in raw_vector]
        if any(not (-1e12 < value < 1e12) for value in vector):
            raise ValueError("vector contains invalid numeric values")
        return vector

    def ensure_ready(self) -> None:
        if not self._ready:
            raise AppError(
                status_code=503,
                code="RAG_INDEX_NOT_READY",
                message=self._not_ready_reason,
            )

    def retrieve(self, *, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        self.ensure_ready()
        if self._vector_dim is None or len(query_vector) != self._vector_dim:
            raise AppError(
                status_code=503,
                code="RAG_INDEX_NOT_READY",
                message="Knowledge base embeddings do not match the loaded index.",
            )

        scored_results: list[RetrievalResult] = []
        for chunk in self._chunks:
            score = self._cosine_similarity(query_vector, chunk.vector)
            if score >= self._score_threshold:
                scored_results.append(RetrievalResult(chunk=chunk, score=score))

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:top_k]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

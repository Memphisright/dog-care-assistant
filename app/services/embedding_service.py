import asyncio
from abc import ABC, abstractmethod

from app.core.exceptions import AppError


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class UnavailableEmbeddingProvider(EmbeddingProvider):
    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def embed_query(self, text: str) -> list[float]:
        _ = text
        raise AppError(
            status_code=503,
            code="RAG_EMBEDDING_UNAVAILABLE",
            message=self._reason,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        raise AppError(
            status_code=503,
            code="RAG_EMBEDDING_UNAVAILABLE",
            message=self._reason,
        )


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise AppError(
                status_code=503,
                code="RAG_EMBEDDING_UNAVAILABLE",
                message=f"Failed to import sentence-transformers: {exc}",
            ) from exc

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            raise AppError(
                status_code=503,
                code="RAG_EMBEDDING_UNAVAILABLE",
                message=f"Failed to load embedding model `{self._model_name}`: {exc}",
            ) from exc
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)


def create_embedding_provider(*, mode: str, model_name: str) -> EmbeddingProvider:
    if mode == "local":
        return LocalEmbeddingProvider(model_name)
    return UnavailableEmbeddingProvider(
        f"Unsupported RAG embedding mode `{mode}`. Supported modes: local."
    )

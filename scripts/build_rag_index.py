import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.services.embedding_service import create_embedding_provider


def _read_markdown_documents(docs_dir: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = _extract_title(text) or path.stem.replace("_", " ").strip().title()
        documents.append(
            {
                "doc_id": path.stem,
                "title": title,
                "source": path.as_posix(),
                "text": text,
            }
        )
    return documents


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _split_markdown_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    sections: list[str] = []
    current_section: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and current_section:
            sections.append("\n".join(current_section).strip())
            current_section = [stripped]
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section).strip())

    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for section in sections:
        normalized = " ".join(section.split())
        if not normalized:
            continue
        if len(normalized) <= chunk_size:
            chunks.append(normalized)
            continue
        for start in range(0, len(normalized), step):
            piece = normalized[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_size >= len(normalized):
                break
    return chunks


async def _build_index() -> Path:
    settings = get_settings()
    docs_dir = Path(settings.rag_docs_dir)
    output_path = Path(settings.rag_index_path)

    documents = _read_markdown_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"No markdown documents found in {docs_dir}.")

    chunk_items: list[dict[str, str]] = []
    texts_to_embed: list[str] = []
    for document in documents:
        chunks = _split_markdown_text(
            document["text"],
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        for index, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{document['doc_id']}#chunk-{index:03d}"
            chunk_items.append(
                {
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
            texts_to_embed.append(chunk_text)

    if not texts_to_embed:
        raise RuntimeError("No chunks were produced from the source markdown documents.")

    embedding_provider = create_embedding_provider(
        mode=settings.rag_embedding_mode,
        model_name=settings.rag_embedding_model,
    )
    vectors = await embedding_provider.embed_documents(texts_to_embed)

    for chunk, vector in zip(chunk_items, vectors, strict=True):
        chunk["vector"] = vector

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_mode": settings.rag_embedding_mode,
        "embedding_model": settings.rag_embedding_model,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
        "chunks": chunk_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal RAG index.")
    parser.parse_args()
    output_path = asyncio.run(_build_index())
    print(f"RAG index written to: {output_path}")


if __name__ == "__main__":
    main()

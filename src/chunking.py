from __future__ import annotations

from .types import Document


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 80
) -> list[Document]:
    chunked_docs: list[Document] = []

    for doc in documents:
        chunks = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = idx
            metadata["total_chunks"] = len(chunks)
            chunked_docs.append(Document(text=chunk, metadata=metadata))

    return chunked_docs

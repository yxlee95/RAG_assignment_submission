from __future__ import annotations

from .types import RetrievedChunk


def build_rag_prompt(user_query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        row = chunk.metadata.get("row_index", "N/A")
        source = chunk.metadata.get("source_file", "unknown")
        context_blocks.append(
            f"[Chunk {i}] source={source}, row={row}, score={chunk.score:.4f}\n{chunk.text}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    return (
        "You are a retrieval-augmented assistant. "
        "Answer only using the provided context. "
        "If the context is insufficient, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {user_query}\n\n"
        "Provide a concise answer, followed by a short bullet list of evidence references "
        "using chunk numbers."
    )

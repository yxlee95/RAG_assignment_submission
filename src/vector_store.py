from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import Document, RetrievedChunk


class NumpyVectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.records: list[dict] = []

    def build(self, vectors: np.ndarray, documents: list[Document]) -> None:
        if len(vectors) != len(documents):
            raise ValueError("Vector and document counts do not match.")
        self.vectors = vectors
        self.records = [
            {
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

    def save(self, vector_path: Path, records_path: Path) -> None:
        if self.vectors is None:
            raise ValueError("Vector store is empty. Build or load before saving.")
        np.save(vector_path, self.vectors)
        with records_path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def load(self, vector_path: Path, records_path: Path) -> None:
        if not vector_path.exists() or not records_path.exists():
            raise FileNotFoundError("Vector index files not found. Build index first.")
        self.vectors = np.load(vector_path)
        with records_path.open("r", encoding="utf-8") as f:
            self.records = json.load(f)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        if self.vectors is None or not self.records:
            raise ValueError("Vector store not initialized.")

        query = query_vector.reshape(1, -1)
        scores = (self.vectors @ query.T).reshape(-1)

        top_indices = np.argsort(-scores)[:top_k]
        results: list[RetrievedChunk] = []
        for idx in top_indices:
            record = self.records[int(idx)]
            results.append(
                RetrievedChunk(
                    text=record["text"],
                    metadata=record["metadata"],
                    score=float(scores[idx]),
                )
            )
        return results

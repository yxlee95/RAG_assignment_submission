from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class _LocalHashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in self._tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], byteorder="little") % self.dim
                vectors[row_idx, index] += 1.0

            norm = np.linalg.norm(vectors[row_idx])
            if norm > 0:
                vectors[row_idx] /= norm

        return vectors


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.backend = "local-hashing"
        self.model: SentenceTransformer | _LocalHashingEmbedder

        if SentenceTransformer is not None:
            local_path = self._find_local_model(model_name)
            if local_path:
                try:
                    self.model = SentenceTransformer(str(local_path))
                    self.backend = "sentence-transformers (local)"
                    print(f"Loaded model from local path: {local_path}")
                    return
                except Exception as exc:
                    print(f"Warning: failed to load local model at {local_path} ({exc}).")

            try:
                self.model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                return
            except Exception as exc:
                print(
                    f"Warning: failed to load '{model_name}' ({exc}). "
                    "Falling back to local hashing embeddings."
                )

        self.model = _LocalHashingEmbedder(dim=384)

    @staticmethod
    def _find_local_model(model_name: str) -> Path | None:
        model_base = model_name.split("/")[-1]
        base_candidates = [
            Path("./models") / model_base,
            Path("models") / model_base,
            Path(model_name),
        ]

        for base_candidate in base_candidates:
            if not base_candidate.exists():
                continue

            nested = base_candidate / model_base
            if nested.exists() and (nested / "pytorch_model.bin").exists():
                return nested

            if (base_candidate / "pytorch_model.bin").exists():
                return base_candidate

            if (base_candidate / "modules.json").exists():
                return base_candidate

        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts)
        return np.asarray(vectors, dtype=np.float32)

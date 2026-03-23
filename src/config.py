from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    storage_dir: Path = project_root / "storage"

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_verify_ssl: bool = os.getenv("GEMINI_VERIFY_SSL", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    gemini_ca_bundle: str = os.getenv("GEMINI_CA_BUNDLE", "")

    top_k: int = int(os.getenv("TOP_K", "4"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))

    index_vectors_path: Path = storage_dir / "vectors.npy"
    index_records_path: Path = storage_dir / "records.json"


def ensure_storage_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

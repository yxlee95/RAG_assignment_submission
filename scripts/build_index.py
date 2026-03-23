from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_documents
from src.config import AppConfig, ensure_storage_dir
from src.embeddings import EmbeddingModel
from src.ingestion import load_documents, resolve_input_file
from src.vector_store import NumpyVectorStore


def main() -> None:
    config = AppConfig()
    ensure_storage_dir(config.storage_dir)

    input_file = resolve_input_file(config.data_dir)
    print(f"[1/4] Loading documents from: {input_file}")
    documents = load_documents(input_file)
    print(f"Loaded rows as documents: {len(documents)}")

    print("[2/4] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=80)
    print(f"Created chunks: {len(chunks)}")

    print("[3/4] Embedding chunks...")
    embedding_model = EmbeddingModel(config.embedding_model)
    texts = [doc.text for doc in tqdm(chunks, desc="Preparing text", leave=False)]
    vectors = embedding_model.encode(texts)

    print("[4/4] Building and saving vector store...")
    store = NumpyVectorStore()
    store.build(vectors=vectors, documents=chunks)
    store.save(config.index_vectors_path, config.index_records_path)

    print("Done.")
    print(f"Saved vectors to: {config.index_vectors_path}")
    print(f"Saved records to: {config.index_records_path}")


if __name__ == "__main__":
    main()

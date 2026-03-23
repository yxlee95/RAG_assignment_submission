from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.create_sample_data import ensure_sample_data

from .types import Document


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def resolve_input_file(data_dir: Path) -> Path:
    preferred = [
        data_dir / "sample_lessons.xlsx",
        data_dir / "sample_lessons.csv",
        data_dir / "Lesson_Learned_Sample.csv",
        data_dir / "Lesson_Learned_Sample.xlsx",
        data_dir / "Lesson_Learned_Sample.xls",
    ]
    for file_path in preferred:
        if file_path.exists():
            return file_path

    sample_file = ensure_sample_data()
    if sample_file.exists():
        return sample_file

    candidates = [p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(
            f"No supported data files found in {data_dir}. Expected CSV or Excel files."
        )
    return sorted(candidates)[0]


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(file_path: Path) -> list[Document]:
    df = _load_dataframe(file_path)
    df = df.fillna("")

    documents: list[Document] = []
    columns = list(df.columns)
    for row_idx, row in df.iterrows():
        parts = [f"{col}: {str(row[col]).strip()}" for col in columns if str(row[col]).strip()]
        text = "\n".join(parts).strip()
        if not text:
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source_file": file_path.name,
                    "row_index": int(row_idx),
                },
            )
        )

    return documents

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

from .types import Document


SUPPORTED_EXTENSIONS = {".docx", ".csv", ".xlsx", ".xls"}


def resolve_input_file(data_dir: Path) -> Path:
    preferred = [
        data_dir / "FAQ.docx",
        data_dir / "sample_lessons.xlsx",
        data_dir / "sample_lessons.csv",
        data_dir / "Lesson_Learned_Sample.csv",
        data_dir / "Lesson_Learned_Sample.xlsx",
        data_dir / "Lesson_Learned_Sample.xls",
    ]
    for file_path in preferred:
        if file_path.exists():
            return file_path

    candidates = [p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(
            f"No supported data files found in {data_dir}. Expected DOCX, CSV, or Excel files."
        )
    return sorted(candidates)[0]


def _load_docx_paragraphs(file_path: Path) -> list[str]:
    with ZipFile(file_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)

    return paragraphs


def _load_docx_documents(file_path: Path) -> list[Document]:
    paragraphs = _load_docx_paragraphs(file_path)
    documents: list[Document] = []

    current_question: str | None = None
    current_answer_lines: list[str] = []
    faq_index = 0

    def flush_pair() -> None:
        nonlocal current_question, current_answer_lines, faq_index
        if not current_question:
            return

        answer_text = " ".join(current_answer_lines).strip()
        full_text = f"Question: {current_question}\nAnswer: {answer_text}".strip()
        documents.append(
            Document(
                text=full_text,
                metadata={
                    "source_file": file_path.name,
                    "faq_index": faq_index,
                    "question": current_question,
                    "answer": answer_text,
                    "faq_full_text": full_text,
                },
            )
        )
        faq_index += 1
        current_question = None
        current_answer_lines = []

    for line in paragraphs:
        normalized = line.strip()
        lower = normalized.lower()

        if lower.startswith("question:"):
            flush_pair()
            current_question = normalized.split(":", 1)[1].strip()
            current_answer_lines = []
            continue

        if lower.startswith("answer:"):
            answer_start = normalized.split(":", 1)[1].strip()
            if answer_start:
                current_answer_lines.append(answer_start)
            continue

        if current_question:
            current_answer_lines.append(normalized)

    flush_pair()
    return documents


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(file_path: Path) -> list[Document]:
    if file_path.suffix.lower() == ".docx":
        return _load_docx_documents(file_path)

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

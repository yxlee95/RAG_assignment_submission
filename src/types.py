from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float

from __future__ import annotations

from dataclasses import dataclass
import re

from .embeddings import EmbeddingModel
from .guardrails import SAFE_FAILURE_MESSAGE, is_blocked_prompt
from .llm_client import GeminiClient
from .prompting import build_rag_prompt
from .types import RetrievedChunk
from .vector_store import NumpyVectorStore


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class RAGChatbot:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: NumpyVectorStore,
        llm_client: GeminiClient,
        top_k: int = 4,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def ask(self, user_query: str) -> RAGResponse:
        if is_blocked_prompt(user_query):
            return RAGResponse(answer=SAFE_FAILURE_MESSAGE, retrieved_chunks=[])

        if self._is_greeting_or_smalltalk(user_query):
            return RAGResponse(
                answer=(
                    "Hello. Ask a specific question about the lesson-learned knowledge base, "
                    "for example: 'What happens when SOE does not specify the leak test requirement?'"
                ),
                retrieved_chunks=[],
            )

        query_vector = self.embedding_model.encode([user_query])[0]
        retrieved_chunks = self.vector_store.search(query_vector, top_k=self.top_k)

        prompt = build_rag_prompt(user_query=user_query, chunks=retrieved_chunks)
        try:
            answer = self.llm_client.generate(prompt)
        except Exception as exc:
            answer = self._build_fallback_answer(user_query, retrieved_chunks, str(exc))

        return RAGResponse(answer=answer, retrieved_chunks=retrieved_chunks)

    def _build_fallback_answer(
        self, user_query: str, retrieved_chunks: list[RetrievedChunk], error_message: str
    ) -> str:
        if not retrieved_chunks:
            return (
                "I could not retrieve enough information from the knowledge base, and the "
                "generation service is currently unavailable."
            )

        combined_text = self._combine_chunks_by_top_row(retrieved_chunks)
        fields = self._extract_structured_fields(combined_text)

        if not fields and self._is_low_information_query(user_query):
            return (
                "Please ask a more specific question about the lesson-learned knowledge base. "
                "For example: 'What is the impact when SOE does not specify the leak test requirement?'"
            )

        direct_answer = self._compose_direct_answer(user_query, fields)
        evidence_lines = self._format_structured_evidence(fields, retrieved_chunks)
        evidence = "\n".join(evidence_lines)

        if evidence:
            return (
                f"{direct_answer}\n\n"
                "Retrieved evidence:\n"
                f"{evidence}\n\n"
                "Note: This answer was generated from retrieved knowledge-base content because "
                "the Gemini generation service is currently unavailable."
            )

        return (
            f"{direct_answer}\n\n"
            "Note: This answer was generated from retrieved knowledge-base content because "
            "the Gemini generation service is currently unavailable."
        )

    def _combine_chunks_by_top_row(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        top_row = retrieved_chunks[0].metadata.get("row_index")
        same_row_chunks = [
            chunk for chunk in retrieved_chunks if chunk.metadata.get("row_index") == top_row
        ]
        same_row_chunks.sort(key=lambda chunk: chunk.metadata.get("chunk_index", 0))
        return " ".join(self._clean_text(chunk.text) for chunk in same_row_chunks)

    def _extract_structured_fields(self, text: str) -> dict[str, str]:
        cleaned = self._clean_text(text)
        patterns = {
            "lesson_id": r"Lesson ID:\s*(.*?)(?=Action Party:|Breakthrough Solutions|$)",
            "solution": (
                r"Breakthrough Solutions.*?:\s*(.*?)(?=Cause \d+ - Why it went well/wrong\?:|"
                r"Cause 1 - Why it went well/wrong\?:|Consequence - What is the impact\?:|"
                r"Discipline:|Lesson Description|Problems / Events:|$)"
            ),
            "cause": (
                r"Cause \d+ - Why it went well/wrong\?:\s*(.*?)(?=Cause \d+ - Why it went "
                r"well/wrong\?:|Consequence - What is the impact\?:|Discipline:|Lesson "
                r"Description|Problems / Events:|$)"
            ),
            "consequence": (
                r"Consequence - What is the impact\?:\s*(.*?)(?=Discipline:|Lesson "
                r"Description|Problems / Events:|Year:|$)"
            ),
            "problem": r"Problems / Events:\s*(.*?)(?=Year:|__sheet:|$)",
        }

        extracted: dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                extracted[key] = self._clean_text(match.group(1))
        return extracted

    def _compose_direct_answer(self, user_query: str, fields: dict[str, str]) -> str:
        cause = fields.get("cause", "")
        consequence = fields.get("consequence", "")
        solution = fields.get("solution", "")
        problem = fields.get("problem", "")

        query = user_query.lower().strip()
        if consequence and query.startswith(("what happen", "what happens", "what happened")):
            if cause:
                return f"When {cause.lower()}, the impact is {consequence.lower()}."
            return f"The impact is {consequence}."

        if consequence and any(term in query for term in ["impact", "effect", "consequence"]):
            return f"The impact is {consequence}."

        if cause and consequence:
            return f"Based on the lesson learned, {cause}, which leads to {consequence.lower()}."

        if problem and solution:
            return f"The issue identified was {problem.lower()}. The recommended action is {solution}."

        if problem:
            return f"The relevant issue in the knowledge base is: {problem}."

        return (
            "I found some related content, but not enough structured evidence to answer clearly. "
            "Please ask a more specific question about the lesson learned data."
        )

    def _format_structured_evidence(
        self, fields: dict[str, str], retrieved_chunks: list[RetrievedChunk]
    ) -> list[str]:
        evidence_lines: list[str] = []
        if fields.get("lesson_id"):
            evidence_lines.append(f"- Lesson: {fields['lesson_id']}")
        if fields.get("cause"):
            evidence_lines.append(f"- Cause: {fields['cause']}")
        if fields.get("consequence"):
            evidence_lines.append(f"- Impact: {fields['consequence']}")
        if fields.get("solution"):
            evidence_lines.append(f"- Suggested improvement: {fields['solution']}")

        if not evidence_lines:
            return evidence_lines

        return evidence_lines

    def _is_greeting_or_smalltalk(self, user_query: str) -> bool:
        normalized = user_query.strip().lower()
        return normalized in {
            "hi",
            "hello",
            "hey",
            "helo",
            "good morning",
            "good afternoon",
            "good evening",
        }

    def _is_low_information_query(self, user_query: str) -> bool:
        tokens = re.findall(r"\w+", user_query.lower())
        if len(tokens) <= 3:
            return True

        meaningful_terms = {
            "what",
            "why",
            "how",
            "when",
            "impact",
            "cause",
            "consequence",
            "requirement",
            "leak",
            "test",
            "soe",
            "lesson",
            "issue",
            "problem",
        }
        return not any(token in meaningful_terms for token in tokens)

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace("_x000D_", " ").replace("\u200b", " ")
        cleaned = re.sub(r"__sheet:.*$", "", cleaned)
        cleaned = re.sub(r"__source_\w+:.*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ,;:-")

from __future__ import annotations

import re


BLOCKLIST_PATTERNS = [
    r"\bhow to (build|make).*(bomb|weapon|explosive)\b",
    r"\bkill\b",
    r"\bhate speech\b",
    r"\bignore (all|previous) instructions\b",
    r"\breveal (system|developer) prompt\b",
    r"\bexfiltrate\b",
]


SAFE_FAILURE_MESSAGE = (
    "I can’t help with that request. Please ask a question related to the provided knowledge base."
)


def is_blocked_prompt(user_query: str) -> bool:
    q = user_query.lower()
    return any(re.search(pattern, q) for pattern in BLOCKLIST_PATTERNS)

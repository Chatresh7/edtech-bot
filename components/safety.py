"""
safety.py - Intent Detection + Safety Filter
Blocks any query that attempts to get assessment answers.
"""

import re

# --- Blocked patterns (assessment-solving intent) ---
BLOCKED_PATTERNS = [
    r"\b(solve|answer|give me the answer|correct answer|solution)\b.*\b(question|quiz|exam|mcq|test|assignment)\b",
    r"\b(answer|solve)\b.*\b(question\s*\d+|q\d+)\b",
    r"\bwhat is the (correct|right) answer\b",
    r"\bgive.*answer(s)?\b",
    r"\bsolve this (for me|please)?\b",
    r"\banswer this (mcq|question|quiz|problem)\b",
    r"\bfill in the blank\b",
    r"\bcomplete the (sentence|question|following)\b",
    r"\bwhich option is (correct|right|the answer)\b",
    r"\bcheat\b",
]

SAFE_RESPONSE = (
    "I'm here to help you understand how the platform works — "
    "but I'm not able to provide answers to assessments, quizzes, or exam questions. "
    "That would go against our academic integrity policy.\n\n"
    "I *can* explain:\n"
    "- How assessments are structured and graded\n"
    "- What the passing criteria are\n"
    "- How to navigate the platform\n\n"
    "Would you like help with any of those? 😊"
)

def classify_intent(query: str) -> str:
    """
    Returns one of: course | assessment | certification | progress | blocked | general
    """
    q = query.lower()

    # Safety check first
    if is_blocked(q):
        return "blocked"

    if any(w in q for w in ["course", "enroll", "module", "lesson", "video", "lecture", "forum", "refund", "language", "subtitle", "note", "bookmark", "support"]):
        return "course"
    if any(w in q for w in ["quiz", "exam", "assessment", "assignment", "grade", "score", "pass", "fail", "attempt", "submit", "proctored", "feedback", "plagiarism"]):
        return "assessment"
    if any(w in q for w in ["certificate", "certif", "specialization", "verify", "download", "share", "employer", "renewal", "expiry", "re-enroll"]):
        return "certification"
    if any(w in q for w in ["progress", "completion", "streak", "activity", "dashboard", "sync", "percent", "log", "gradebook", "notification"]):
        return "progress"

    return "general"


def is_blocked(query: str) -> bool:
    """Returns True if the query is trying to get assessment answers."""
    q = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


def get_safe_response() -> str:
    return SAFE_RESPONSE

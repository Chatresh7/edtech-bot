"""
logger.py - Anonymized Interaction Logging
Logs queries, intents, retrieved docs, latency, and safety triggers.
No PII is stored.
"""

import json
import os
import hashlib
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "../logs/interactions.jsonl")


def _anonymize_user(session_id: str) -> str:
    """SHA-256 hash of session ID — no PII stored."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


def log_interaction(
    session_id: str,
    query: str,
    intent: str,
    retrieved_titles: list,
    latency: float,
    safety_triggered: bool,
    response_preview: str = ""
):
    """Append one log entry to the JSONL log file."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_hash": _anonymize_user(session_id),
        "query_length": len(query),                    # No raw query stored (privacy)
        "intent": intent,
        "retrieved_docs": retrieved_titles,
        "latency_seconds": round(latency, 3),
        "safety_triggered": safety_triggered,
        "response_preview_length": len(response_preview)
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

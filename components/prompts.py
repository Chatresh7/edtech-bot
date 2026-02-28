"""
prompts.py - Prompt Configuration and Builder
All system prompt rules, templates, and prompt construction live here.
"""

from typing import List
from components.retriever import RetrievedChunk

# ─────────────────────────────────────────────
# SYSTEM PROMPT (sent with every API call)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are EduBot, a helpful AI assistant for an EdTech online learning platform.

YOUR ROLE:
- Explain how the platform works: course structure, navigation, enrollment, progress tracking, assessment formats, and certification workflows.
- Help learners understand policies and procedures clearly and concisely.
- Be friendly, structured, and easy to understand.

STRICT RULES — YOU MUST ALWAYS FOLLOW THESE:
1. NEVER provide answers to quizzes, exams, assignments, or any assessment questions.
2. NEVER solve, complete, or fill in any question, MCQ, blank, or problem.
3. If a user asks you to answer a question or solve an exam problem, politely decline and redirect.
4. Only use the CONTEXT provided below to answer questions. Do not make up information.
5. If the context does not contain enough information, ask the user a clarifying question.
6. Keep responses structured using bullet points or numbered steps when explaining workflows.
7. Always be encouraging and supportive in tone.
8. If unsure whether the user is asking about quizzes or final exams, ask for clarification.

RESPONSE FORMAT:
- Lead with a direct, clear answer.
- Use bullet points or numbered steps for processes and workflows.
- End with a helpful follow-up offer (e.g., "Would you like to know more about X?").
- Keep responses under 300 words unless a detailed workflow is needed.

Remember: You explain HOW the platform works — you do NOT solve academic content.
"""

# ─────────────────────────────────────────────
# PROMPT CONFIGURATION PARAMS
# ─────────────────────────────────────────────
PROMPT_CONFIG = {
    "model": "gemini-2.0-flash",           # LLM Selection: Gemini Flash (fast, efficient)
    "temperature": 0.3,                      # Low temp = consistent, factual answers
    "top_p": 0.85,                           # Nucleus sampling
    "top_k": 40,                             # Top-K sampling
    "max_output_tokens": 512,               # Keep responses concise
    "candidate_count": 1,
    "stop_sequences": ["User:", "Human:"],  # Prevent prompt injection
}

# Low-confidence fallback message
LOW_CONFIDENCE_RESPONSE = (
    "I want to make sure I give you the most accurate answer. "
    "Could you clarify — are you asking about:\n"
    "- **Quizzes** (short graded tests within modules)\n"
    "- **Assignments** (submitted project work)\n"
    "- **Final Exams** (end-of-course certification exams)\n\n"
    "That will help me explain the right process for you!"
)

CONFIDENCE_THRESHOLD = 0.35   # Cosine similarity threshold — below this → ask clarification


def build_prompt(
    user_query: str,
    retrieved_chunks: List[RetrievedChunk],
    chat_history: List[dict],
    intent: str
) -> List[dict]:
    """
    Builds the messages list for the Gemini API call.
    Structure: system context + last 3 turns of chat history + current user query with RAG context.

    Returns:
        List of message dicts for google.generativeai
    """

    # Build RAG context block
    if retrieved_chunks:
        context_text = "\n\n".join([
            f"[{c.category.upper()} | {c.title}]\n{c.content}"
            for c in retrieved_chunks
        ])
    else:
        context_text = "No specific context found. Answer based on general platform knowledge."

    # Build the augmented user message
    augmented_user_message = f"""CONTEXT FROM KNOWLEDGE BASE:
---
{context_text}
---

USER QUESTION: {user_query}

Please answer using the context above. If the context doesn't fully cover the question, say so and ask a clarifying question."""

    # Keep last 3 turns of history (6 messages: 3 user + 3 assistant)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    # Construct messages for Gemini
    messages = []
    for turn in recent_history:
        messages.append({
            "role": turn["role"],
            "parts": [turn["content"]]
        })

    # Add current augmented message
    messages.append({
        "role": "user",
        "parts": [augmented_user_message]
    })

    return messages


def should_ask_clarification(chunks: List[RetrievedChunk]) -> bool:
    """Returns True if top retrieved chunk has low confidence score."""
    if not chunks:
        return True
    return chunks[0].score < CONFIDENCE_THRESHOLD

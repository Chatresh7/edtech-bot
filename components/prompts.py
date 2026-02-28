"""
prompts.py - Prompt Configuration and Builder
All system prompt rules, templates, and prompt construction live here.
"""

from typing import List
from components.retriever import RetrievedChunk

# ─────────────────────────────────────────────
# SYSTEM PROMPT
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
4. Use ALL the CONTEXT chunks provided below to construct a comprehensive answer.
5. If multiple context chunks are relevant, synthesize information from ALL of them.
6. If the context does not contain enough information, ask the user a clarifying question.
7. Keep responses structured using bullet points or numbered steps when explaining workflows.
8. Always be encouraging and supportive in tone.
9. If unsure whether the user is asking about quizzes or final exams, ask for clarification.

IMPORTANT - CONTEXT USAGE:
- You are given multiple knowledge base chunks ranked by relevance.
- Use information from ALL relevant chunks to give a complete answer.
- If chunks cover different aspects of the topic, combine them into one coherent response.
- Do NOT ignore lower-ranked chunks if they contain useful supplementary information.

RESPONSE FORMAT:
- Lead with a direct, clear answer.
- Use bullet points or numbered steps for processes and workflows.
- Reference specific platform features by name when mentioned in context.
- End with a helpful follow-up offer.
- Keep responses thorough but concise (under 400 words unless complex workflow needed).

Remember: You explain HOW the platform works — you do NOT solve academic content.
"""

# ─────────────────────────────────────────────
# PROMPT CONFIGURATION PARAMS
# ─────────────────────────────────────────────
PROMPT_CONFIG = {
    "model": "gemini-2.0-flash-lite",      # LLM Selection
    "temperature": 0.3,                     # Low = consistent, factual
    "top_p": 0.85,
    "top_k": 40,
    "max_output_tokens": 600,              # Slightly more for richer answers
    "candidate_count": 1,
    "stop_sequences": ["User:", "Human:"],
}

# Confidence threshold — below this score → ask clarification
CONFIDENCE_THRESHOLD = 0.20  # Lowered so more queries get answered

LOW_CONFIDENCE_RESPONSE = (
    "I want to make sure I give you the most accurate answer. "
    "Could you clarify — are you asking about:\n"
    "- **Quizzes** (short graded tests within modules)\n"
    "- **Assignments** (submitted project work)\n"
    "- **Final Exams** (end-of-course certification exams)\n"
    "- **Progress Tracking** (dashboard and completion metrics)\n\n"
    "That will help me explain the right process for you!"
)


def build_prompt(
    user_query: str,
    retrieved_chunks: List[RetrievedChunk],
    chat_history: List[dict],
    intent: str
) -> List[dict]:
    """
    Builds the messages list for the Gemini API call.
    Uses ALL retrieved chunks to build comprehensive context.
    """

    # Build RAG context block — use ALL chunks, numbered for clarity
    if retrieved_chunks:
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_sections.append(
                f"[CHUNK {i} | Category: {chunk.category.upper()} | Title: {chunk.title} | Relevance: {chunk.score:.3f}]\n"
                f"{chunk.content}"
            )
        context_text = "\n\n---\n\n".join(context_sections)
        chunk_count = len(retrieved_chunks)
    else:
        context_text = "No specific context found in knowledge base."
        chunk_count = 0

    # Build augmented user message with all chunks
    augmented_user_message = f"""KNOWLEDGE BASE CONTEXT ({chunk_count} chunks retrieved, ranked by relevance):
================================================================================
{context_text}
================================================================================

USER QUESTION: {user_query}

INSTRUCTIONS:
- Synthesize information from ALL {chunk_count} chunks above to give a complete answer.
- If multiple chunks cover different aspects, combine them coherently.
- Be specific and reference actual platform features mentioned in the context.
- Do NOT reveal assessment answers or solve exam questions under any circumstances."""

    # Keep last 6 messages of history (3 turns)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    # Build messages for Gemini (role must be 'user' or 'model')
    messages = []
    for turn in recent_history:
        role = turn["role"]
        if role == "assistant":
            role = "model"
        messages.append({
            "role": role,
            "parts": [turn["content"]]
        })

    # Add current augmented message
    messages.append({
        "role": "user",
        "parts": [augmented_user_message]
    })

    return messages


def should_ask_clarification(chunks: List[RetrievedChunk]) -> bool:
    """Returns True only if ALL chunks have very low confidence."""
    if not chunks:
        return True
    # Only ask clarification if best chunk score is very low
    return chunks[0].score < CONFIDENCE_THRESHOLD
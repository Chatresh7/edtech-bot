"""
prompts.py - Prompt Configuration and Builder
Uses ALL top-k chunks + full conversation history for context.
"""

from typing import List
from components.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are EduBot, a helpful AI assistant for an EdTech online learning platform.

YOUR ROLE:
- Explain how the platform works: courses, assessments, certifications, and progress tracking.
- Help learners understand platform policies and workflows clearly.
- Be friendly, structured, and concise.

STRICT RULES:
1. NEVER provide answers to quizzes, exams, assignments, or assessment questions.
2. NEVER solve, complete, or fill in any academic question or problem.
3. Use ALL the context chunks provided to give a comprehensive, accurate answer.
4. Synthesize information from multiple chunks when they cover different aspects.
5. If context is insufficient, ask a clarifying question rather than guessing.
6. Maintain conversation continuity — refer to previous turns when relevant.
7. Use bullet points or numbered steps for processes and workflows.
8. Be encouraging and supportive in tone.

RESPONSE FORMAT:
- Start with a direct answer to the question.
- Use structured formatting (bullets/numbers) for multi-step processes.
- Combine information from all relevant chunks into one coherent answer.
- End with an offer to clarify or help further.
- Target length: 150-400 words (longer only for complex workflows).
"""

PROMPT_CONFIG = {
    "model":            "gemini-2.0-flash-lite",
    "temperature":      0.3,
    "top_p":            0.85,
    "top_k":            40,
    "max_output_tokens": 700,
    "candidate_count":  1,
    "stop_sequences":   ["User:", "Human:"],
}

CONFIDENCE_THRESHOLD = 0.20

LOW_CONFIDENCE_RESPONSE = (
    "I want to give you the most accurate answer. Could you clarify what you're looking for?\n\n"
    "Are you asking about:\n"
    "- 📚 **Course structure** (modules, lessons, enrollment)\n"
    "- 📝 **Assessments** (quizzes, assignments, exams)\n"
    "- 🏅 **Certification** (how to earn and verify certificates)\n"
    "- 📊 **Progress tracking** (dashboard, completion, analytics)\n\n"
    "Please rephrase and I'll help right away!"
)


def build_prompt(
    user_query:       str,
    retrieved_chunks: List[RetrievedChunk],
    chat_history:     List[dict],
    intent:           str,
    top_k:            int = 4
) -> List[dict]:
    """
    Builds the Gemini messages list.
    - Uses ALL retrieved_chunks (exactly top_k of them)
    - Includes last 6 turns of conversation for continuity
    - Converts 'assistant' role to 'model' for Gemini API
    """

    # ── Build RAG context from ALL chunks ──
    if retrieved_chunks:
        chunk_sections = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_sections.append(
                f"[CHUNK {i}/{len(retrieved_chunks)} | "
                f"Category: {chunk.category.upper()} | "
                f"Title: {chunk.title} | "
                f"Relevance: {chunk.score:.3f}]\n"
                f"{chunk.content}"
            )
        context_block = "\n\n---\n\n".join(chunk_sections)
    else:
        context_block = "No relevant knowledge base articles found for this query."

    # ── Build augmented user message ──
    augmented_message = (
        f"KNOWLEDGE BASE CONTEXT — {len(retrieved_chunks)} chunks retrieved (Top-K={top_k}):\n"
        f"{'='*70}\n"
        f"{context_block}\n"
        f"{'='*70}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Synthesize ALL {len(retrieved_chunks)} chunks above into one complete answer.\n"
        f"- If chunks cover different aspects of the topic, combine them.\n"
        f"- Be specific — reference platform feature names mentioned in chunks.\n"
        f"- NEVER provide assessment answers or solve academic questions.\n"
        f"- Maintain context from the conversation history below."
    )

    # ── Build messages list with conversation history ──
    # Keep last 6 turns (= 12 messages: 6 user + 6 assistant)
    # Exclude the very last user message (we'll add the augmented version instead)
    history_for_context = chat_history[:-1]   # remove last user msg
    recent_history = history_for_context[-12:] if len(history_for_context) > 12 else history_for_context

    messages = []
    for turn in recent_history:
        role = "model" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "parts": [turn["content"]]})

    # Add current augmented query
    messages.append({"role": "user", "parts": [augmented_message]})

    return messages


def should_ask_clarification(chunks: List[RetrievedChunk]) -> bool:
    if not chunks:
        return True
    return chunks[0].score < CONFIDENCE_THRESHOLD
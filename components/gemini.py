"""
gemini.py - Gemini Flash API Integration
Handles API calls, response validation, and post-response safety checks.
"""

import os
import re
import time
import google.generativeai as genai
from components.prompts import SYSTEM_PROMPT, PROMPT_CONFIG

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────
# Model Selection: gemini-2.0-flash
# Rationale: Fast (<2s), cost-efficient, supports long context, ideal for FAQ/workflow bots
LLM_MODEL = "gemini-2.5-flash-lite"

# Patterns that would indicate answer leakage in the response
ANSWER_LEAKAGE_PATTERNS = [
    r"\bthe (correct|right) answer is\b",
    r"\boption [a-d] is correct\b",
    r"\banswer[:=]\s*[a-d]\b",
    r"\b(question \d+)[:=]",
    r"\bthe solution is\b",
]

LEAKAGE_BLOCK_RESPONSE = (
    "I noticed my response might have included assessment-specific information "
    "that I should not share. I've blocked that response to maintain academic integrity.\n\n"
    "I can still help you understand **how assessments work** on our platform. "
    "Would you like me to explain the assessment format or grading policy instead?"
)


def init_gemini():
    """Initialize Gemini with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your environment variables or .env file.\n"
            "Get your key from: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)


def get_gemini_model():
    """Returns a configured GenerativeModel instance."""
    generation_config = genai.GenerationConfig(
        temperature=PROMPT_CONFIG["temperature"],
        top_p=PROMPT_CONFIG["top_p"],
        top_k=PROMPT_CONFIG["top_k"],
        max_output_tokens=PROMPT_CONFIG["max_output_tokens"],
        candidate_count=PROMPT_CONFIG["candidate_count"],
        stop_sequences=PROMPT_CONFIG["stop_sequences"],
    )
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT
    )
    return model


def validate_response(response_text: str) -> tuple[bool, str]:
    """
    Post-response validation: checks for answer leakage.
    Returns (is_safe: bool, response_or_block_message: str)
    """
    text_lower = response_text.lower()
    for pattern in ANSWER_LEAKAGE_PATTERNS:
        if re.search(pattern, text_lower):
            return False, LEAKAGE_BLOCK_RESPONSE
    return True, response_text


def call_gemini(messages: list, retry: int = 2) -> tuple[str, float]:
    """
    Calls the Gemini Flash API.
    Args:
        messages: list of {role, parts} dicts
        retry: number of retries on rate limit
    Returns:
        (response_text, latency_seconds)
    """
    model = get_gemini_model()

    # Convert message list to Gemini chat history format
    history = messages[:-1]   # All except last
    last_message = messages[-1]["parts"][0]

    for attempt in range(retry + 1):
        try:
            start = time.time()
            chat = model.start_chat(history=history)
            response = chat.send_message(last_message)
            latency = time.time() - start
            raw_text = response.text

            # Post-response validation
            is_safe, final_text = validate_response(raw_text)
            return final_text, latency

        except Exception as e:
            if attempt < retry and "429" in str(e):
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e

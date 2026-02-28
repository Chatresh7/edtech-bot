"""
gemini.py - Gemini Flash API Integration
Handles API calls, role conversion, response validation.
"""

import os
import re
import time
import google.generativeai as genai
from components.prompts import SYSTEM_PROMPT, PROMPT_CONFIG

LLM_MODEL = PROMPT_CONFIG["model"]

ANSWER_LEAKAGE_PATTERNS = [
    r"\bthe (correct|right) answer is\b",
    r"\boption [a-d] is correct\b",
    r"\banswer[:=]\s*[a-d]\b",
    r"\bthe solution is\b",
    r"\bcorrect option is\b",
]

LEAKAGE_BLOCK = (
    "I noticed my response may have included assessment-specific content I should not share.\n\n"
    "I can explain **how assessments work** on the platform instead. "
    "Would you like me to explain the assessment format or grading policy?"
)


def init_gemini():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in Streamlit Secrets or your .env file.\n"
            "Get your key at: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)


def get_model():
    config = genai.GenerationConfig(
        temperature      = PROMPT_CONFIG["temperature"],
        top_p            = PROMPT_CONFIG["top_p"],
        top_k            = PROMPT_CONFIG["top_k"],
        max_output_tokens= PROMPT_CONFIG["max_output_tokens"],
        candidate_count  = PROMPT_CONFIG["candidate_count"],
        stop_sequences   = PROMPT_CONFIG["stop_sequences"],
    )
    return genai.GenerativeModel(
        model_name        = LLM_MODEL,
        generation_config = config,
        system_instruction= SYSTEM_PROMPT
    )


def validate_response(text: str):
    """Post-response safety check for answer leakage."""
    lower = text.lower()
    for pattern in ANSWER_LEAKAGE_PATTERNS:
        if re.search(pattern, lower):
            return False, LEAKAGE_BLOCK
    return True, text


def call_gemini(messages: list, retries: int = 2):
    """
    Calls Gemini API.
    - Converts 'assistant' → 'model' for Gemini compatibility
    - Sends full conversation history for continuity
    - Returns (response_text, latency_seconds)
    """
    model = get_model()

    # Fix roles: Gemini uses 'user' and 'model' (NOT 'assistant')
    fixed = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        fixed.append({"role": role, "parts": msg["parts"]})

    # Split into history (all but last) and current message
    history      = fixed[:-1]
    last_message = fixed[-1]["parts"][0]

    for attempt in range(retries + 1):
        try:
            t0       = time.time()
            chat     = model.start_chat(history=history)
            response = chat.send_message(last_message)
            latency  = time.time() - t0

            _, final_text = validate_response(response.text)
            return final_text, latency

        except Exception as e:
            err_str = str(e)
            if attempt < retries and ("429" in err_str or "500" in err_str):
                time.sleep(2 ** attempt)   # exponential backoff
                continue
            # Friendly fallback instead of raw error
            return (
                "⚠️ I'm temporarily unavailable due to a technical issue. "
                "Please try again in a moment or contact **support@edtech.com** for help.",
                0.0
            )
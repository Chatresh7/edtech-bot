"""
app.py - EdTech Platform Explainer Bot (Project 42)
Main Streamlit application entry point.

Run with: streamlit run app.py
"""

import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Component imports
from components.safety import classify_intent, is_blocked, get_safe_response
from components.retriever import get_retriever
from components.prompts import build_prompt, should_ask_clarification, LOW_CONFIDENCE_RESPONSE
from components.gemini import init_gemini, call_gemini
from components.logger import log_interaction

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduBot — Course Explainer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.95rem; }

    .intent-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-course        { background:#dbeafe; color:#1d4ed8; }
    .badge-assessment    { background:#fef3c7; color:#b45309; }
    .badge-certification { background:#d1fae5; color:#065f46; }
    .badge-progress      { background:#ede9fe; color:#5b21b6; }
    .badge-general       { background:#f3f4f6; color:#374151; }
    .badge-blocked       { background:#fee2e2; color:#b91c1c; }

    .source-card {
        background: #f8fafc;
        border-left: 3px solid #667eea;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
    .quick-btn { margin: 0.25rem; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "last_reset" not in st.session_state:
    st.session_state.last_reset = time.time()
if "retriever" not in st.session_state:
    with st.spinner("Loading Knowledge Base and building FAISS index..."):
        st.session_state.retriever = get_retriever()
if "gemini_ready" not in st.session_state:
    try:
        init_gemini()
        st.session_state.gemini_ready = True
    except ValueError as e:
        st.session_state.gemini_ready = False
        st.session_state.gemini_error = str(e)

RATE_LIMIT = 10      # requests per minute
RATE_WINDOW = 60     # seconds


def check_rate_limit() -> bool:
    now = time.time()
    if now - st.session_state.last_reset > RATE_WINDOW:
        st.session_state.request_count = 0
        st.session_state.last_reset = now
    if st.session_state.request_count >= RATE_LIMIT:
        return False
    st.session_state.request_count += 1
    return True


INTENT_BADGE = {
    "course":        ("📚 Course",        "badge-course"),
    "assessment":    ("📝 Assessment",    "badge-assessment"),
    "certification": ("🏅 Certification", "badge-certification"),
    "progress":      ("📊 Progress",      "badge-progress"),
    "general":       ("💬 General",       "badge-general"),
    "blocked":       ("🚫 Blocked",       "badge-blocked"),
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 EduBot Settings")

    # API Key input if not set
    if not st.session_state.get("gemini_ready"):
        st.markdown("#### 🔑 Gemini API Key")
        api_key_input = st.text_input("Enter your Gemini API Key", type="password", help="Get from https://aistudio.google.com")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            try:
                init_gemini()
                st.session_state.gemini_ready = True
                st.success("✅ API Key accepted!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.success("✅ Gemini API Connected")
        st.caption(f"Model: **gemini-2.0-flash**")

    st.divider()
    st.markdown("#### ⚙️ RAG Settings")
    top_k = st.slider("Top-K Chunks", 2, 6, 4, help="Number of KB articles retrieved per query")

    st.divider()
    st.markdown("#### 📖 Knowledge Base")
    kb_stats = {
        "course": 14, "assessment": 12,
        "certification": 10, "progress": 10,
        "Total": 50
    }
    for cat, count in kb_stats.items():
        icon = {"course":"📚","assessment":"📝","certification":"🏅","progress":"📊","Total":"🗂️"}.get(cat,"•")
        st.markdown(f"{icon} **{cat.title()}**: {count} articles")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("📞 Escalate to Support", use_container_width=True):
        st.info("Redirecting to Help Desk...\n\n📧 support@edtech-platform.com\n💬 Live chat: 9am–6pm")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.caption(f"Requests this minute: {st.session_state.request_count}/{RATE_LIMIT}")

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 EduBot — Course & Learning Workflow Explainer</h1>
    <p>Ask me anything about course navigation, assessments, certification workflows, or progress tracking.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUICK ACTION BUTTONS
# ─────────────────────────────────────────────
st.markdown("**Quick Questions:**")
cols = st.columns(4)
quick_questions = [
    ("📚 Course Structure",   "How is a course structured on this platform?"),
    ("📝 Assessment Types",   "What types of assessments are there?"),
    ("🏅 Get Certificate",    "How do I earn a certificate?"),
    ("📊 Track Progress",     "How can I track my course progress?"),
]

if "quick_query" not in st.session_state:
    st.session_state.quick_query = None

for col, (label, question) in zip(cols, quick_questions):
    with col:
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state.quick_query = question

# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "intent" in msg:
                label, css = INTENT_BADGE.get(msg["intent"], ("💬","badge-general"))
                st.markdown(f'<span class="intent-badge {css}">{label}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-card">🗂️ <b>{src["title"]}</b> — <i>{src["category"]}</i> (score: {src["score"]:.2f})</div>', unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("latency"):
                st.caption(f"⚡ Response in {msg['latency']:.2f}s")

# ─────────────────────────────────────────────
# PROCESS QUERY (chat input OR quick button)
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask about courses, assessments, certifications, or progress tracking...")

# Use quick query if set
active_query = st.session_state.quick_query or user_input
if st.session_state.quick_query:
    st.session_state.quick_query = None

if active_query:
    query = active_query.strip()
    if not query:
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Rate limit check
    if not check_rate_limit():
        st.warning("⏳ Rate limit reached (10 requests/min). Please wait a moment.")
        st.stop()

    # Safety check
    safety_triggered = is_blocked(query)
    intent = classify_intent(query)

    with st.chat_message("assistant"):
        label, css = INTENT_BADGE.get(intent, ("💬","badge-general"))
        st.markdown(f'<span class="intent-badge {css}">{label}</span>', unsafe_allow_html=True)

        if safety_triggered:
            response = get_safe_response()
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", "content": response,
                "intent": "blocked", "sources": [], "latency": 0
            })
            log_interaction(st.session_state.session_id, query, "blocked", [], 0, True, response)

        elif not st.session_state.get("gemini_ready"):
            st.error("⚠️ Gemini API not configured. Please enter your API key in the sidebar.")

        else:
            with st.spinner("Searching knowledge base and generating response..."):
                # RAG Retrieval
                t0 = time.time()
                chunks = st.session_state.retriever.retrieve(query, category_filter=intent, top_k=top_k)
                retrieval_latency = time.time() - t0

                # Check confidence
                if should_ask_clarification(chunks):
                    response = LOW_CONFIDENCE_RESPONSE
                    latency = time.time() - t0
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response,
                        "intent": intent, "sources": [], "latency": latency
                    })
                else:
                    # Build prompt
                    messages = build_prompt(
                        user_query=query,
                        retrieved_chunks=chunks,
                        chat_history=[m for m in st.session_state.messages if m["role"] in ("user","assistant")],
                        intent=intent
                    )

                    # Call Gemini
                    try:
                        response, latency = call_gemini(messages)
                        st.markdown(response)

                        sources = [{"title": c.title, "category": c.category, "score": c.score} for c in chunks]
                        with st.expander("📎 Sources used", expanded=False):
                            for src in sources:
                                st.markdown(f'<div class="source-card">🗂️ <b>{src["title"]}</b> — <i>{src["category"]}</i> (score: {src["score"]:.2f})</div>', unsafe_allow_html=True)

                        st.caption(f"⚡ Response in {latency:.2f}s | 📚 {len(chunks)} sources retrieved ({retrieval_latency*1000:.0f}ms)")

                        st.session_state.messages.append({
                            "role": "assistant", "content": response,
                            "intent": intent, "sources": sources, "latency": latency
                        })
                        log_interaction(
                            st.session_state.session_id, query, intent,
                            [c.title for c in chunks], latency, False, response
                        )

                    except Exception as e:
                        err_msg = f"⚠️ Error calling Gemini API: {str(e)}\n\nPlease check your API key and try again."
                        st.error(err_msg)
                        st.session_state.messages.append({
                            "role": "assistant", "content": err_msg,
                            "intent": intent, "sources": [], "latency": 0
                        })

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.8rem;'>"
    "🎓 EduBot — Project 42 | EdTech Platform Explainer | "
    "Powered by Gemini Flash + FAISS RAG | "
    "Academic integrity enforced — assessment answers are never provided."
    "</div>",
    unsafe_allow_html=True
)

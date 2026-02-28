"""
app.py - EdTech Platform Explainer Bot (Project 42)
Fixes: conversation continuity, top_k properly passed to retriever and prompt
"""

import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from components.safety import classify_intent, is_blocked, get_safe_response
from components.retriever import get_retriever
from components.prompts import build_prompt, should_ask_clarification, LOW_CONFIDENCE_RESPONSE
from components.gemini import init_gemini, call_gemini
from components.logger import log_interaction

st.set_page_config(
    page_title="EduBot — Course Explainer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.95rem; }
    .intent-badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-course        { background:#dbeafe; color:#1d4ed8; }
    .badge-assessment    { background:#fef3c7; color:#b45309; }
    .badge-certification { background:#d1fae5; color:#065f46; }
    .badge-progress      { background:#ede9fe; color:#5b21b6; }
    .badge-general       { background:#f3f4f6; color:#374151; }
    .badge-blocked       { background:#fee2e2; color:#b91c1c; }
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #667eea;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #e2e8f0 !important;
    }
    .source-card b {
        color: #a5b4fc !important;
    }
    .source-card i {
        color: #94a3b8 !important;
    }
    .chunk-info {
        background: #eff6ff; border-radius: 6px;
        padding: 0.3rem 0.8rem; font-size: 0.78rem; color: #1d4ed8; margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
if "session_id"       not in st.session_state: st.session_state.session_id       = str(uuid.uuid4())
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []   # for Gemini API (role+content only)
if "display_messages" not in st.session_state: st.session_state.display_messages = []   # for UI rendering (full data)
if "request_count"    not in st.session_state: st.session_state.request_count    = 0
if "last_reset"       not in st.session_state: st.session_state.last_reset       = time.time()
if "quick_query"      not in st.session_state: st.session_state.quick_query      = None

if "retriever" not in st.session_state:
    with st.spinner("⏳ Loading Knowledge Base and building FAISS index..."):
        st.session_state.retriever = get_retriever()

if "gemini_ready" not in st.session_state:
    try:
        init_gemini()
        st.session_state.gemini_ready = True
    except ValueError as e:
        st.session_state.gemini_ready = False
        st.session_state.gemini_error = str(e)

RATE_LIMIT  = 10
RATE_WINDOW = 60

INTENT_BADGE = {
    "course":        ("📚 Course",        "badge-course"),
    "assessment":    ("📝 Assessment",    "badge-assessment"),
    "certification": ("🏅 Certification", "badge-certification"),
    "progress":      ("📊 Progress",      "badge-progress"),
    "general":       ("💬 General",       "badge-general"),
    "blocked":       ("🚫 Blocked",       "badge-blocked"),
}

def check_rate_limit():
    now = time.time()
    if now - st.session_state.last_reset > RATE_WINDOW:
        st.session_state.request_count = 0
        st.session_state.last_reset    = now
    if st.session_state.request_count >= RATE_LIMIT:
        return False
    st.session_state.request_count += 1
    return True

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🎓 EduBot Settings")

    if not st.session_state.get("gemini_ready"):
        st.markdown("#### 🔑 Gemini API Key")
        api_key_input = st.text_input("Enter your Gemini API Key", type="password")
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
        st.success("✅ Gemini Connected")
        st.caption("Model: **gemini-2.5-flash-lite**")

    st.divider()
    st.markdown("#### ⚙️ RAG Settings")

    # TOP-K SLIDER — stored in session_state so it persists across reruns
    top_k = st.slider(
        "Top-K Chunks", min_value=1, max_value=8, value=4, step=1,
        help="How many KB articles to retrieve. More chunks = more context = richer answers."
    )
    st.session_state.top_k = top_k   # ← save to session_state
    st.info(f"📦 **{top_k} chunks** will be retrieved per query")

    st.divider()
    st.markdown("#### 📖 Knowledge Base")
    try:
        stats   = st.session_state.retriever.get_stats()
        icon_map = {"course":"📚","assessment":"📝","certification":"🏅","progress":"📊"}
        for cat, count in stats["by_category"].items():
            st.markdown(f"{icon_map.get(cat,'•')} **{cat.title()}**: {count} articles")
        st.markdown(f"🗂️ **Total**: {stats['total_articles']} articles")
    except Exception:
        st.markdown("🗂️ **Total**: 100 articles")

    st.divider()
    st.markdown("#### 💬 Conversation")
    turns = len([m for m in st.session_state.display_messages if m["role"] == "user"])
    st.caption(f"Turns: **{turns}** | Context window: last **6 turns**")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history     = []
            st.session_state.display_messages = []
            st.session_state.quick_query      = None
            st.rerun()
    with c2:
        if st.button("📞 Support", use_container_width=True):
            st.info("📧 chatreshkonchada@gmail.com\n💬 9am–6pm")

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.caption(f"Requests/min: {st.session_state.request_count}/{RATE_LIMIT}")

# ── HEADER ──
st.markdown("""
<div class="main-header">
    <h1>🎓 EduBot — Course & Learning Workflow Explainer</h1>
    <p>Ask me anything about course navigation, assessments, certification workflows, or progress tracking.</p>
</div>
""", unsafe_allow_html=True)

# ── QUICK BUTTONS ──
st.markdown("**Quick Questions:**")
cols = st.columns(4)
quick_questions = [
    ("📚 Course Structure",  "How is a course structured on this platform?"),
    ("📝 Assessment Types",  "What types of assessments are there and how are they graded?"),
    ("🏅 Get Certificate",   "How do I earn a certificate and what are the eligibility requirements?"),
    ("📊 Track Progress",    "How can I track my course progress and view my completion percentage?"),
]
for col, (label, question) in zip(cols, quick_questions):
    with col:
        if st.button(label, use_container_width=True, key=f"qbtn_{label}"):
            st.session_state.quick_query = question

# ── RENDER CHAT HISTORY ──
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            badge_label, badge_css = INTENT_BADGE.get(msg.get("intent","general"), ("💬","badge-general"))
            st.markdown(f'<span class="intent-badge {badge_css}">{badge_label}</span>', unsafe_allow_html=True)
            chunk_count = msg.get("chunk_count", 0)
            if chunk_count > 0:
                st.markdown(
                    f'<div class="chunk-info">🔍 <b>{chunk_count} KB chunks</b> retrieved (Top-K={chunk_count})</div>',
                    unsafe_allow_html=True
                )

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} sources used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card"><b>Chunk {i}:</b> 🗂️ <b>{src["title"]}</b>'
                        f' — <i>{src["category"].title()}</i> | Relevance: <b>{src["score"]:.3f}</b></div>',
                        unsafe_allow_html=True
                    )

        if msg["role"] == "assistant" and msg.get("latency", 0) > 0:
            st.caption(f"⚡ {msg['latency']:.2f}s")

# ── CHAT INPUT ──
user_input = st.chat_input("Ask about courses, assessments, certifications, or progress tracking...")

active_query = st.session_state.quick_query or user_input
if st.session_state.quick_query:
    st.session_state.quick_query = None

if active_query:
    query = active_query.strip()
    if not query:
        st.stop()

    # Read top_k from session_state (set by slider above)
    current_top_k = st.session_state.get("top_k", 4)

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Add to both histories
    st.session_state.display_messages.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Rate limit
    if not check_rate_limit():
        with st.chat_message("assistant"):
            st.warning("⏳ Rate limit reached (10/min). Please wait.")
        st.stop()

    safety_triggered = is_blocked(query)
    intent           = classify_intent(query)

    with st.chat_message("assistant"):
        badge_label, badge_css = INTENT_BADGE.get(intent, ("💬", "badge-general"))
        st.markdown(f'<span class="intent-badge {badge_css}">{badge_label}</span>', unsafe_allow_html=True)

        # ── BLOCKED ──
        if safety_triggered:
            response = get_safe_response()
            st.markdown(response)
            st.session_state.display_messages.append({
                "role":"assistant","content":response,
                "intent":"blocked","sources":[],"chunk_count":0,"latency":0
            })
            st.session_state.chat_history.append({"role":"assistant","content":response})
            log_interaction(st.session_state.session_id, query, "blocked", [], 0, True, response)

        elif not st.session_state.get("gemini_ready"):
            st.error("⚠️ Gemini API not configured. Please enter your API key in the sidebar.")

        else:
            # ── STEP 1: RETRIEVE exactly current_top_k chunks ──
            with st.spinner(f"🔍 Retrieving top {current_top_k} knowledge base chunks..."):
                t0 = time.time()
                chunks = st.session_state.retriever.retrieve(
                    query=query,
                    category_filter=intent,
                    top_k=current_top_k     # ← EXACTLY what slider says
                )
                retrieval_ms = (time.time() - t0) * 1000

            if should_ask_clarification(chunks):
                response = LOW_CONFIDENCE_RESPONSE
                st.markdown(response)
                st.session_state.display_messages.append({
                    "role":"assistant","content":response,
                    "intent":intent,"sources":[],"chunk_count":0,"latency":0
                })
                st.session_state.chat_history.append({"role":"assistant","content":response})

            else:
                # Show retrieval info
                st.markdown(
                    f'<div class="chunk-info">🔍 Retrieved <b>{len(chunks)} chunks</b> '
                    f'(Top-K={current_top_k}) in {retrieval_ms:.0f}ms</div>',
                    unsafe_allow_html=True
                )

                # ── STEP 2: BUILD PROMPT with ALL chunks + conversation history ──
                messages_for_gemini = build_prompt(
                    user_query=query,
                    retrieved_chunks=chunks,
                    chat_history=st.session_state.chat_history,  # full conversation
                    intent=intent,
                    top_k=current_top_k
                )

                # ── STEP 3: CALL GEMINI ──
                try:
                    with st.spinner("🤖 Generating response..."):
                        response, latency = call_gemini(messages_for_gemini)

                    st.markdown(response)

                    sources = [{"title":c.title,"category":c.category,"score":c.score} for c in chunks]
                    with st.expander(f"📎 {len(sources)} sources used", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-card"><b>Chunk {i}:</b> 🗂️ <b>{src["title"]}</b>'
                                f' — <i>{src["category"].title()}</i> | Relevance: <b>{src["score"]:.3f}</b></div>',
                                unsafe_allow_html=True
                            )

                    conv_turns = len(st.session_state.chat_history) // 2
                    st.caption(
                        f"⚡ {latency:.2f}s | 📚 {len(chunks)}/{current_top_k} chunks | "
                        f"🔍 {retrieval_ms:.0f}ms retrieval | 💬 {conv_turns} turns in context"
                    )

                    # Save to display_messages (UI)
                    st.session_state.display_messages.append({
                        "role":"assistant","content":response,
                        "intent":intent,"sources":sources,
                        "chunk_count":len(chunks),"latency":latency
                    })

                    # Save to chat_history (Gemini conversation continuity)
                    st.session_state.chat_history.append({"role":"assistant","content":response})

                    log_interaction(
                        st.session_state.session_id, query, intent,
                        [c.title for c in chunks], latency, False, response
                    )

                except Exception as e:
                    err = f"⚠️ Error calling Gemini API: {str(e)}\n\nPlease check your API key."
                    st.error(err)
                    st.session_state.display_messages.append({
                        "role":"assistant","content":err,
                        "intent":intent,"sources":[],"chunk_count":0,"latency":0
                    })
                    st.session_state.chat_history.append({"role":"assistant","content":err})

# ── FOOTER ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
    "🎓 EduBot — Project 42 | Gemini Flash + FAISS RAG | "
    "Academic integrity enforced — assessment answers are never provided."
    "</div>", unsafe_allow_html=True
)
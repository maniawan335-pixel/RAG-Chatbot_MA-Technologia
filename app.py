import os
import base64
import markdown as md_lib
import streamlit as st
from dotenv import load_dotenv

# Project modules — DO NOT MODIFY
from vector_store import get_or_create_vector_store
from rag_pipeline import get_rag_chain, HybridMemory

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CAIRO – MA Technologia",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Logo helper — encode image to base64 for inline HTML display
# ---------------------------------------------------------------------------
def get_logo_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

LOGO_PATH = os.path.join(os.path.dirname(__file__), "MA TECH LOGO.png")
logo_b64 = get_logo_base64(LOGO_PATH)
logo_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else ""

# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Orbitron:wght@700;900&display=swap');

/* ── Base & background ───────────────────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020b18 0%, #050505 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

/* Remove Streamlit header bar */
header[data-testid="stHeader"] { display: none !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #040e1c !important;
    border-right: 1px solid #1a2d45 !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] input {
    background: #0d1f35 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Main container ──────────────────────────────────────────────────── */
.block-container {
    padding: 0 1rem 2rem 1rem !important;
    max-width: 820px !important;
}

/* ── Header card ─────────────────────────────────────────────────────── */
.cairo-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
    background: linear-gradient(180deg, #061020 0%, transparent 100%);
    border-bottom: 1px solid #1a2d45;
    margin-bottom: 1.5rem;
}
.cairo-logo {
    width: 160px;
    height: auto;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px rgba(74,144,217,0.4));
}
.cairo-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 0.25em;
    color: #ffffff;
    text-shadow: 0 0 30px rgba(74,144,217,0.7), 0 0 60px rgba(77,240,192,0.2);
    margin: 0;
    line-height: 1;
}
.cairo-subtitle {
    font-size: 0.9rem;
    color: #4A90D9;
    letter-spacing: 0.15em;
    font-weight: 500;
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.cairo-tagline {
    font-size: 0.75rem;
    color: #4df0c0;
    letter-spacing: 0.2em;
    margin-top: 0.25rem;
    opacity: 0.8;
}
.cairo-divider {
    border: none;
    border-top: 1px solid #1a2d45;
    margin: 1.2rem auto 0 auto;
    width: 60%;
    opacity: 0.6;
}

/* ── Chat messages — override Streamlit defaults ─────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Custom bubble wrappers ──────────────────────────────────────────── */
.chat-wrap { display: flex; flex-direction: column; margin-bottom: 1.2rem; }

.cairo-label {
    font-size: 0.65rem;
    color: #4A90D9;
    letter-spacing: 0.15em;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    margin-left: 2px;
}

.user-bubble {
    align-self: flex-end;
    background: #4A90D9;
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    line-height: 1.6;
    font-size: 0.92rem;
    box-shadow: 0 4px 20px rgba(74,144,217,0.25);
}

.cairo-bubble {
    align-self: flex-start;
    background: #0d1f35;
    color: #cdd9e8;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    line-height: 1.7;
    font-size: 0.92rem;
    border-left: 3px solid #4A90D9;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.cairo-bubble p { margin: 0 0 0.5em 0; }
.cairo-bubble p:last-child { margin-bottom: 0; }
.cairo-bubble ul, .cairo-bubble ol { margin: 0.4em 0 0.4em 1.2em; }
.cairo-bubble li { margin-bottom: 0.2em; }

/* Welcome bubble special highlight */
.welcome-bubble {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1929 100%);
    border-left: 3px solid #4df0c0;
    box-shadow: 0 4px 30px rgba(77,240,192,0.1);
}

/* ── Typing indicator ────────────────────────────────────────────────── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 14px 18px;
    background: #0d1f35;
    border-radius: 4px 18px 18px 18px;
    border-left: 3px solid #4A90D9;
    width: fit-content;
}
.typing-dot {
    width: 8px;
    height: 8px;
    background: #4A90D9;
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 60%, 100% { transform: scale(0.7); opacity: 0.4; }
    30% { transform: scale(1); opacity: 1; }
}

/* ── Chat input area ─────────────────────────────────────────────────── */
[data-testid="stBottom"] > div {
    background: linear-gradient(0deg, #020b18 0%, transparent 100%) !important;
    padding: 1rem 0 !important;
}
[data-testid="stChatInput"] {
    background: #0d1f35 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    box-shadow: 0 0 20px rgba(74,144,217,0.08) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4A90D9 !important;
    box-shadow: 0 0 25px rgba(74,144,217,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    caret-color: #4A90D9 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #3a5a7a !important;
}
[data-testid="stChatInput"] button {
    background: #4A90D9 !important;
    border-radius: 8px !important;
    color: #fff !important;
}
[data-testid="stChatInput"] button:hover {
    background: #357bb5 !important;
}
[data-testid="stChatInput"] button svg {
    fill: #fff !important;
    stroke: #fff !important;
}

/* ── Clear button ────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #1e3a5f !important;
    color: #4A90D9 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #0d1f35 !important;
    border-color: #4A90D9 !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #4A90D9 !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020b18; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4A90D9; }

/* ── Info/warning messages ───────────────────────────────────────────── */
.stAlert { background: #0d1f35 !important; border: 1px solid #1e3a5f !important; color: #e2e8f0 !important; }

/* ── Sidebar branding footer ─────────────────────────────────────────── */
.sidebar-brand {
    position: fixed;
    bottom: 1.5rem;
    font-size: 0.7rem;
    color: #1e3a5f !important;
    letter-spacing: 0.1em;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend initialisation — untouched logic
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.markdown("""
    <div style="color:#4A90D9;font-size:0.75rem;letter-spacing:0.1em;margin-bottom:0.5rem;">
    🔑 API KEY REQUIRED
    </div>""", unsafe_allow_html=True)
    api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")

if not api_key:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#3a5a7a;">
        <div style="font-family:Orbitron,monospace;font-size:1.5rem;color:#4A90D9;margin-bottom:0.5rem;">CAIRO</div>
        <div style="font-size:0.9rem;">Please provide a Groq API key in the sidebar to start.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

DATA_DIR = "./data"
INDEX_PATH = "./faiss_index"

@st.cache_resource(show_spinner=False)
def load_vector_store(key: str):
    return get_or_create_vector_store(DATA_DIR, INDEX_PATH, key)

vector_store = load_vector_store(api_key)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = HybridMemory(window_size=4)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
if logo_src:
    logo_html = f'<img src="{logo_src}" class="cairo-logo" alt="MA Technologia Logo">'
else:
    logo_html = ""

st.markdown(f"""
<div class="cairo-header">
    {logo_html}
    <div class="cairo-title">CAIRO</div>
    <div class="cairo-subtitle">MA Technologia Virtual Assistant</div>
    <div class="cairo-tagline">Build. Scale. Automate.</div>
    <hr class="cairo-divider">
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="color:#4A90D9;font-size:0.65rem;letter-spacing:0.2em;font-weight:700;
         text-transform:uppercase;margin-bottom:1.5rem;padding-top:0.5rem;">
    ⚙ CONTROLS
    </div>""", unsafe_allow_html=True)

    if st.button("🗑 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;background:#040e1c;border:1px solid #1a2d45;
         border-radius:8px;font-size:0.75rem;color:#3a5a7a;line-height:1.7;">
        <div style="color:#4A90D9;font-weight:700;letter-spacing:0.1em;margin-bottom:0.5rem;">ABOUT CAIRO</div>
        CAIRO is MA Technologia's virtual assistant, trained on our services,
        projects, and expertise. Ask about web development, AI solutions, pricing,
        or anything related to our agency.
    </div>
    <div class="sidebar-brand" style="color:#1e3a5f;font-size:0.68rem;
         letter-spacing:0.12em;margin-top:3rem;text-align:center;">
        © 2024 MA TECHNOLOGIA
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Welcome message (shown when no messages yet)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-wrap">
        <div class="cairo-label">CAIRO</div>
        <div class="cairo-bubble welcome-bubble">
            Assalam o Alaikum! 👋 I'm <strong style="color:#4df0c0;">CAIRO</strong>,
            MA Technologia's virtual assistant.<br><br>
            I can help you learn about our services, ongoing projects, pricing,
            and how we can build your next digital solution.<br><br>
            <span style="color:#4A90D9;">How can I help you today?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"""
        <div class="chat-wrap">
            <div class="user-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        html_content = md_lib.markdown(content, extensions=["nl2br"])
        st.markdown(f"""
        <div class="chat-wrap">
            <div class="cairo-label">CAIRO</div>
            <div class="cairo-bubble">{html_content}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input & response
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask CAIRO anything...")

if prompt:
    # Show user bubble immediately
    st.markdown(f"""
    <div class="chat-wrap">
        <div class="user-bubble">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.memory.add_message("user", prompt)

    # Typing indicator + spinner
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="chat-wrap">
        <div class="cairo-label">CAIRO</div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        chain, _, llm = get_rag_chain(vector_store)
        chat_history = st.session_state.memory.get_history_text()
        summary = st.session_state.memory.summary

        result = chain({
            "question": prompt,
            "chat_history": chat_history,
            "summary": summary
        })
        answer = result.get("answer", "")

        # Clear typing indicator and show actual response
        typing_placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.memory.add_message("ai", answer)
        st.session_state.memory.update_summary(llm)

    except Exception as e:
        typing_placeholder.empty()
        st.error(f"CAIRO encountered an error: {e}")

    st.rerun()

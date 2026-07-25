"""
Streamlit chat UI. A thin client over the FastAPI service — it holds the
conversation and the selected source, and does no retrieval itself.

Run locally (with the API already running on :8000):
    streamlit run streamlit_app.py
"""

import base64
import os
from pathlib import Path

import requests
import streamlit as st

# When API_URL is set the UI talks to the FastAPI service over HTTP. When it
# isn't — as on HuggingFace Spaces, which runs a single process — the UI calls
# rag_core in-process instead. Same UI either way.
API_URL = os.getenv("API_URL")
DIRECT_MODE = not API_URL
TIMEOUT = 120

if DIRECT_MODE:
    import rag_core

BACKGROUND = Path(__file__).parent / "assets" / "background.jpg"
BACKGROUND_B64 = Path(__file__).parent / "assets" / "background.jpg.b64"

STARTERS = [
    "Who was Odysseus?",
    "Tell me about Perseus and Medusa",
    "How did Zeus come to power?",
    "What happened on Circe's island?",
]

st.set_page_config(
    page_title="Greek Mythology RAG Agent",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- styling

def background_css():
    """
    Full-bleed background image, encoded inline so no static file server is
    needed. Falls back to a gradient when the image hasn't been added.
    """

    if BACKGROUND.exists():
        data = base64.b64encode(BACKGROUND.read_bytes()).decode()
    elif BACKGROUND_B64.exists():
        # HuggingFace rejects binary files that aren't in LFS, so the deployed
        # copy ships pre-encoded as text. It goes straight into the data URI.
        data = BACKGROUND_B64.read_text().strip()
    else:
        data = None

    if data:
        layer = f'url("data:image/jpeg;base64,{data}")'
    else:
        layer = "linear-gradient(160deg, #2b2f4a 0%, #4a3f55 45%, #1c1a26 100%)"

    return f"""
    .stApp {{
        background: {layer};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """


def load_styles():

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

        {background_css()}

        /* Scrim so light areas of the painting stay readable, kept light
           enough that the image is still clearly visible. */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse at 50% 38%, rgba(8,6,16,0.12) 0%, rgba(8,6,16,0.58) 100%),
                linear-gradient(180deg, rgba(10,8,20,0.30) 0%, rgba(10,8,20,0.50) 100%);
            pointer-events: none;
            z-index: 0;
        }}

        .stApp > * {{ position: relative; z-index: 1; }}

        /* Streamlit paints opaque panels over the background; clear them so
           the image runs edge to edge behind the glass. */
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        section.main {{
            background: transparent !important;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, sans-serif;
        }}

        /* ---------------- header ---------------- */

        .hero {{ text-align: center; padding: 0.5rem 0 1.75rem; }}

        .hero h1 {{
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 3.1rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #f6efe2;
            margin: 0;
            text-shadow: 0 2px 24px rgba(0,0,0,0.65);
        }}

        .hero p {{
            font-size: 0.94rem;
            font-weight: 300;
            color: rgba(246,239,226,0.72);
            margin: 0.6rem auto 0;
            max-width: 34rem;
            line-height: 1.6;
        }}

        .rule {{
            width: 78px;
            height: 1px;
            margin: 1.1rem auto 0;
            background: linear-gradient(90deg, transparent, rgba(212,175,110,0.9), transparent);
        }}

        /* ---------------- glass panels ---------------- */

        [data-testid="stChatMessage"] {{
            background: rgba(255,255,255,0.07);
            backdrop-filter: blur(16px) saturate(150%);
            -webkit-backdrop-filter: blur(16px) saturate(150%);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.32);
            color: #f2ece1;
        }}

        /* User turns tinted warm so the two speakers are distinguishable
           without relying on avatars alone. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background: rgba(212,175,110,0.13);
            border-color: rgba(212,175,110,0.28);
        }}

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {{
            color: #f2ece1;
            line-height: 1.68;
            font-size: 0.95rem;
        }}

        [data-testid="stChatMessage"] strong {{ color: #f0d9a8; }}

        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3 {{
            font-family: 'Cormorant Garamond', Georgia, serif;
            color: #f6efe2;
            font-size: 1.3rem;
            margin-top: 0.4rem;
        }}

        /* ---------------- sidebar ---------------- */

        /* The colour sits on the sidebar element itself, so it has to be
           overridden there rather than on an inner wrapper. */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(16,14,26,0.42) !important;
            backdrop-filter: blur(26px) saturate(140%);
            -webkit-backdrop-filter: blur(26px) saturate(140%);
            border-right: 1px solid rgba(255,255,255,0.10);
        }}

        [data-testid="stSidebar"] * {{ color: #ece5d8 !important; }}

        [data-testid="stSidebar"] h2 {{
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 1.5rem;
            letter-spacing: 0.03em;
        }}

        /* ---------------- chat input ---------------- */

        [data-testid="stChatInput"] {{
            background: rgba(255,255,255,0.09);
            backdrop-filter: blur(18px) saturate(150%);
            -webkit-backdrop-filter: blur(18px) saturate(150%);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px;
        }}

        [data-testid="stChatInput"] textarea {{ color: #f4eee3 !important; }}

        [data-testid="stChatInput"] textarea::placeholder {{
            color: rgba(244,238,227,0.45) !important;
        }}

        /* ---------------- buttons ---------------- */

        .stButton > button {{
            background: rgba(255,255,255,0.07);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 11px;
            color: #ece5d8;
            font-size: 0.86rem;
            font-weight: 400;
            padding: 0.55rem 0.9rem;
            transition: all 0.18s ease;
            width: 100%;
        }}

        .stButton > button:hover {{
            background: rgba(212,175,110,0.20);
            border-color: rgba(212,175,110,0.55);
            color: #fdf6e8;
            transform: translateY(-1px);
        }}

        /* ---------------- misc ---------------- */

        [data-testid="stCaptionContainer"] p {{
            color: rgba(236,229,216,0.58) !important;
            font-size: 0.79rem;
        }}

        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}

        [data-testid="stSpinner"] > div {{ border-top-color: #d4af6e !important; }}

        .empty-hint {{
            text-align: center;
            color: rgba(246,239,226,0.55);
            font-size: 0.87rem;
            margin: 0.5rem 0 1.1rem;
            letter-spacing: 0.02em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


load_styles()


# ---------------------------------------------------------------- data

@st.cache_resource(show_spinner="Preparing the texts — this takes a moment on first load...")
def bootstrap():
    """
    Build the vector store from sources/ if it's empty. Lets a fresh deploy
    ship only the ~3MB of text rather than the ~32MB prebuilt index.
    """

    if rag_core.list_sources():
        return

    folder = Path(__file__).parent / "sources"

    for path in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.pdf")):
        rag_core.add_source(str(path))


@st.cache_data(ttl=30)
def get_health():

    if DIRECT_MODE:
        try:
            return {
                "status": "ok" if rag_core.chunk_count() else "empty",
                "model": rag_core.MODEL,
                "chunks": rag_core.chunk_count(),
                "sources": rag_core.list_sources(),
            }, None
        except Exception as exc:
            return None, str(exc)

    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def ask_agent(question, source, history):
    """
    Returns (answer, tool_called). Routes to the API or straight to rag_core.
    """

    if DIRECT_MODE:
        called = []
        answer = rag_core.ask(
            question=question,
            source=source,
            history=history,
            on_tool_call=lambda name, scope: called.append(name),
        )
        return answer, bool(called)

    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "source": source, "history": history},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["answer"], data["tool_called"]


if DIRECT_MODE:
    bootstrap()

health, error = get_health()


# ---------------------------------------------------------------- sidebar

with st.sidebar:

    st.header("Sources")

    if error:

        # The API spends ~40s loading the embedding model on boot, so a
        # refused connection is usually "still starting", not "broken".
        st.warning("Waiting for the API")
        st.caption(
            f"If you just started it, it takes about 40 seconds to load the "
            f"embedding model. Otherwise, check that it's running on {API_URL}."
        )

        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()

        with st.expander("Error detail"):
            st.caption(error)

        st.stop()

    sources = health.get("sources", [])

    if not sources:
        st.warning("No texts are loaded.")
        st.stop()

    choice = st.radio(
        "Search in",
        ["All sources"] + sources,
        help="Restrict retrieval to a single text, or search across everything.",
        format_func=lambda s: s if s == "All sources" else s.replace(".txt", "").replace("_", " ").title(),
    )

    st.divider()

    st.caption(f"{health['chunks']:,} passages indexed")
    st.caption(f"{len(sources)} texts · {health['model']}")

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if not BACKGROUND.exists() and not BACKGROUND_B64.exists():
        st.caption("Tip: add assets/background.jpg for the full theme.")

selected_source = None if choice == "All sources" else choice


# ---------------------------------------------------------------- header

st.markdown(
    """
    <div class="hero">
        <h1>Greek Mythology</h1>
        <p>Answers drawn only from the loaded public-domain texts —
           Homer, Hesiod, Bulfinch, Kingsley and Hawthorne.</p>
        <div class="rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []


def send(question):
    """
    Queue a question and rerun so it renders through the normal flow.
    """
    st.session_state.pending = question
    st.rerun()


# Starter prompts, shown only on an empty conversation — they teach what the
# agent is good at without the user having to guess.
if not st.session_state.messages:

    st.markdown('<div class="empty-hint">Try one of these</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    for i, starter in enumerate(STARTERS):
        with (left if i % 2 == 0 else right):
            if st.button(starter, key=f"starter_{i}"):
                send(starter)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])

typed = st.chat_input("Ask about the gods, heroes or monsters...")

question = typed or st.session_state.pop("pending", None)

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Consulting the texts..."):

            try:
                answer, tool_called = ask_agent(
                    question=question,
                    source=selected_source,
                    # Send prior turns so follow-ups like "what about her
                    # island?" resolve; exclude the turn just added.
                    history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ],
                )
                caption = (
                    f"Searched {selected_source or 'all sources'}"
                    if tool_called
                    else "Answered without searching"
                )

            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                caption = None

        st.markdown(answer)

        if caption:
            st.caption(caption)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "caption": caption}
    )

    st.rerun()

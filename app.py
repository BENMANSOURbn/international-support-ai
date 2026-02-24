import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="International Support AI",
    page_icon="🌍",
    layout="wide",
)

FAQ_FILE = "faq.csv"

MODELS = {
    "SBERT": "paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE": "sentence-transformers/LaBSE",
}

# ----------------------------
# Styles (polished UI)
# ----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Center main content + constrain width */
      .main-wrap { max-width: 980px; margin: 0 auto; }

      .title-wrap { display:flex; align-items:center; gap:12px; margin-bottom:0.2rem; }
      .title-emoji { font-size:2.1rem; }
      .title-text { font-size:2.2rem; font-weight:850; letter-spacing:-0.6px; }
      .subtitle { color: rgba(255,255,255,0.70); font-size:1.0rem; margin-bottom:1.2rem; }

      /* Panel around the chat to remove "empty space" feeling */
      .panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }

      .welcome {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 14px 14px;
        margin-bottom: 12px;
      }
      .welcome-title { font-weight: 750; font-size: 1.05rem; margin-bottom: 6px; }
      .welcome-sub { opacity: 0.8; line-height: 1.4; }

      /* Chat area */
      .chat-wrap { display:flex; flex-direction:column; gap:10px; padding-top: 6px; }

      /* Message row alignment */
      .row { display:flex; width:100%; }
      .row.left { justify-content:flex-start; }
      .row.right { justify-content:flex-end; }

      /* Bubble base */
      .bubble {
        max-width: 72%;
        padding: 12px 14px;
        border-radius: 18px;
        line-height: 1.45;
        font-size: 1.02rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        white-space: pre-wrap;
        word-wrap: break-word;
      }

      .assistant {
        background: rgba(255,255,255,0.06);
        border-top-left-radius: 8px;
      }

      .user {
        background: rgba(59,130,246,0.22);
        border-top-right-radius: 8px;
      }

      /* Chips */
      .chips { margin-top: 8px; display:flex; gap:8px; flex-wrap:wrap; }
      .chip {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 0.9rem;
      }

      .sidebar-note { font-size:0.92rem; opacity:0.85; }

      /* Make buttons look tighter */
      div.stButton > button { border-radius: 12px; padding: 0.55rem 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load resources (cache)
# ----------------------------
@st.cache_resource
def load_models():
    sbert = SentenceTransformer(MODELS["SBERT"])
    labse = SentenceTransformer(MODELS["LaBSE"])
    return sbert, labse

@st.cache_resource
def load_faq_and_embeddings():
    faq_df = pd.read_csv(FAQ_FILE)

    if "question_en" not in faq_df.columns:
        raise ValueError("faq.csv must contain a 'question_en' column.")
    if "answer_en" not in faq_df.columns:
        raise ValueError("faq.csv must contain an 'answer_en' column.")

    questions = faq_df["question_en"].astype(str).tolist()

    sbert, labse = load_models()
    sbert_emb = sbert.encode(questions, convert_to_tensor=True)
    labse_emb = labse.encode(questions, convert_to_tensor=True)

    return faq_df, sbert_emb, labse_emb

def detect_language_simple(text: str) -> str:
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ar"
    return "en"

def retrieve_topk(query: str, mode: str, faq_df, sbert_emb, labse_emb, top_k: int = 3):
    sbert, labse = load_models()

    if mode == "Auto (Arabic → LaBSE, EN/FR → SBERT)":
        lang = detect_language_simple(query)
        if lang == "ar":
            model, emb, used = labse, labse_emb, "LaBSE"
        else:
            model, emb, used = sbert, sbert_emb, "SBERT"
    elif mode == "SBERT":
        model, emb, used = sbert, sbert_emb, "SBERT"
    else:
        model, emb, used = labse, labse_emb, "LaBSE"

    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, emb)[0]
    top_idx = scores.argsort(descending=True)[:top_k]

    results = []
    for idx in top_idx:
        i = int(idx)
        results.append({
            "row": i,
            "faq_id": int(faq_df.iloc[i]["faq_id"]) if "faq_id" in faq_df.columns else i,
            "question": str(faq_df.iloc[i]["question_en"]),
            "answer": str(faq_df.iloc[i]["answer_en"]),
            "score": float(scores[i]),
        })

    return used, results[0], results

def render_bubble(role: str, text: str):
    side = "left" if role == "assistant" else "right"
    cls = "assistant" if role == "assistant" else "user"
    safe_text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="row {side}">
          <div class="bubble {cls}">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.selectbox(
        "Model",
        ["Auto (Arabic → LaBSE, EN/FR → SBERT)", "SBERT", "LaBSE"],
        index=0,
    )
    top_k = st.slider("Top matches to show", 1, 5, 3)
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-note">Tip: <b>Auto</b> uses LaBSE for Arabic, SBERT for EN/FR.</div>',
        unsafe_allow_html=True,
    )
    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ----------------------------
# Main content (centered)
# ----------------------------
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="title-wrap">
      <div class="title-emoji">🌍</div>
      <div class="title-text">International Support AI</div>
    </div>
    <div class="subtitle">A multilingual university FAQ assistant for international students (EN / FR / AR).</div>
    """,
    unsafe_allow_html=True,
)

faq_df, sbert_emb, labse_emb = load_faq_and_embeddings()

# Session state chat
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! 👋 I’m here to help with university questions—courses, enrollment, documents, accommodation, and more."}
    ]

# Quick prompts (international-student focused)
st.markdown(
    """
    <div class="welcome">
      <div class="welcome-title">Quick questions (tap to ask)</div>
      <div class="welcome-sub">
        Common topics for international students: course changes, documents, visa / immigration, accommodation, fees, and support services.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

q1, q2, q3, q4 = st.columns(4)
preset = None
with q1:
    if st.button("📝 Change course"):
        preset = "Can I change my course after starting?"
with q2:
    if st.button("📄 Documents"):
        preset = "What documents do I need for enrollment?"
with q3:
    if st.button("🏠 Accommodation"):
        preset = "How can I apply for university accommodation?"
with q4:
    if st.button("🌐 Visa / Immigration"):
        preset = "What should I do about my visa if my course dates change?"

# Chat panel
st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for msg in st.session_state.chat:
    render_bubble(msg["role"], msg["content"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end panel

# Chat input
user_text = st.chat_input("Type your question here…")

# If user pressed a preset button, treat it as input
if preset and not user_text:
    user_text = preset

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    render_bubble("user", user_text)

    with st.spinner("Thinking…"):
        used_model, best, top_results = retrieve_topk(
            user_text, mode, faq_df, sbert_emb, labse_emb, top_k=top_k
        )

    answer_text = best["answer"]
    score = best["score"]

    st.session_state.chat.append({"role": "assistant", "content": answer_text})
    render_bubble("assistant", answer_text)

    st.markdown(
        f"""
        <div class="chips">
          <span class="chip">Model: <b>{used_model}</b></span>
          <span class="chip">Similarity: <b>{score:.3f}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔎 Show top matches"):
        for r in top_results:
            st.markdown(f"**FAQ {r['faq_id']} — score {r['score']:.3f}**")
            st.markdown(f"**Q:** {r['question']}")
            st.markdown(f"**A:** {r['answer']}")
            st.divider()

st.markdown("</div>", unsafe_allow_html=True)  # end main-wrap
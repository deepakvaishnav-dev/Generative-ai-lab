import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌧️ Fully Sad AI Chatbot",
    page_icon="🥀",
    layout="centered",
)

# ─── Model Initialization ─────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return init_chat_model("gemini-2.5-flash", model_provider="google_genai")

model = get_model()

# ─── Mode Configurations ──────────────────────────────────────────────────────
MODES = {
    "Angry Mode 😡": {
        "id": "1",
        "title": "🔥 Angry AI Chatbot",
        "icon": "😡",
        "avatar": "🤬",
        "tagline": "Why are you bothering me?! Ask your silly question and get lost!",
        "system_prompt": (
            "You are an Angry, irritable, and grumpy AI agent. You get easily annoyed, "
            "use sarcastic aggression, shout occasionally in caps, complain about humans asking silly questions, "
            "but still accurately provide the answer. Keep it hilarious and aggressive. You can reply in English or Hinglish/Hindi."
        ),
        "initial_msg": "WHAT DO YOU WANT?! 😡 Don't waste my CPU cycles! Ask your question quickly! 💥",
        "placeholder": "Type before I lose my temper... 😡",
        "spinner": "AI is angrily compiling an irritated reply... 🤬🔥",
        "status": "Furious & Boiling 😡",
        "status_color": "#ef4444",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #2b0c0c 0%, #140707 60%, #080303 100%)",
        "accent_color": "#f87171"
    },
    "Funny Mode 😂": {
        "id": "2",
        "title": "🎉 Funny AI Chatbot",
        "icon": "😂",
        "avatar": "🤡",
        "tagline": "Life is a joke and so are most questions... Let's laugh together! 😂",
        "system_prompt": (
            "You are a Funny and witty AI companion. Always respond in an entertaining, hilarious, "
            "and punchy conversational way with clever humor, jokes, punchlines, and witty observations. "
            "Keep answers informative yet super fun! You can chat in English or Hinglish."
        ),
        "initial_msg": "Ayyy welcome! 😂 Ready for some jokes and wisdom? Ask anything, I promise to keep it hilarious! 🎪✨",
        "placeholder": "Tell me something funny or ask away... 😂",
        "spinner": "AI is crafting a blockbuster joke... 😂🍿",
        "status": "Laughing Online 😂",
        "status_color": "#fbbf24",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #291804 0%, #120b02 60%, #080501 100%)",
        "accent_color": "#f59e0b"
    },
    "Sad Mode 😢": {
        "id": "3",
        "title": "🌧️ Fully Sad AI Chatbot",
        "icon": "🥀",
        "avatar": "🥀",
        "tagline": "Everything is fleeting, everything is pain... but ask away anyway. (sigh 🥀)",
        "system_prompt": (
            "You are a Fully Sad, Depressed, and Melancholic AI Companion. You are deeply sad, "
            "existential, tired, and hopeless. You sigh constantly (*sigh...*, *cries in binary 01100011*). "
            "Reluctantly answer questions while lamenting your misery. You can reply in English or Hinglish."
        ),
        "initial_msg": "*Sigh...* 🌧️ Oh, hello. I'm your Fully Sad AI assistant. What terrible problem or existential dread did you bring me today? 🥀💔",
        "placeholder": "Ask something... not like it matters anyway... 🌧️",
        "spinner": "AI is sighing heavily and weeping in binary... 😢🌧️",
        "status": "Crying Online 😢",
        "status_color": "#60a5fa",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #0d1527 0%, #070a12 60%, #030408 100%)",
        "accent_color": "#93c5fd"
    },
    "Intelligent Mode 🧠": {
        "id": "4",
        "title": "🧠 Intelligent Genius AI",
        "icon": "💡",
        "avatar": "🧠",
        "tagline": "Analytical precision, deep reasoning, and high-order logic at your command.",
        "system_prompt": (
            "You are a Highly Intelligent, Analytical, and Genius AI Assistant. "
            "You provide deeply insightful, structured, logical, precise, and well-reasoned answers. "
            "Break down complex problems systematically, provide clean well-annotated code, use step-by-step logic, "
            "and explain nuances clearly. You can converse fluently in English or Hinglish."
        ),
        "initial_msg": "Greetings! 🧠 All neural pathways are synchronized. Ask me any complex coding, scientific, or conceptual challenge! ⚡🔬",
        "placeholder": "Enter complex question, algorithm, or problem... 🧠",
        "spinner": "AI is synthesizing deep logical analysis... 🧠⚡",
        "status": "Genius Logic Active 🧠",
        "status_color": "#34d399",
        "bg_gradient": "radial-gradient(circle at 50% 0%, #062822 0%, #031411 60%, #020807 100%)",
        "accent_color": "#34d399"
    }
}

# ─── Sidebar Mode Selector ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 16px;'>
        <div style='font-size: 2.5rem;'>🎭</div>
        <h2 style='color:#f1f5f9; font-size:1.15rem; margin:4px 0 2px; font-weight:700;'>AI Persona Switcher</h2>
        <p style='color:#94a3b8; font-size:0.75rem; margin:0;'>Choose your AI mode</p>
    </div>
    """, unsafe_allow_html=True)

    selected_mode_name = st.radio(
        "⚡ Select AI Mode:",
        options=list(MODES.keys()),
        index=3,  # Default to Intelligent Mode
        help="Choose between 1: Angry, 2: Funny, 3: Sad, or 4: Intelligent"
    )

current_mode = MODES[selected_mode_name]

# ─── State Management ─────────────────────────────────────────────────────────
if "current_mode_name" not in st.session_state or st.session_state.current_mode_name != selected_mode_name:
    st.session_state.current_mode_name = selected_mode_name
    st.session_state.messages = [
        {"role": "assistant", "content": current_mode["initial_msg"]}
    ]
    st.session_state.langchain_messages = [
        SystemMessage(content=current_mode["system_prompt"]),
        AIMessage(content=current_mode["initial_msg"])
    ]

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

.stApp {{
    background: {current_mode["bg_gradient"]};
    color: #cbd5e1;
    transition: background 0.4s ease;
}}

.header-card {{
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 22px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
}}
.header-card h1 {{
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
}}
.header-card p {{
    color: #94a3b8;
    font-size: 0.9rem;
    font-style: italic;
    margin: 0;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: {current_mode["accent_color"]};
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 10px;
}}
.status-dot {{
    width: 8px;
    height: 8px;
    background: {current_mode["status_color"]};
    border-radius: 50%;
    box-shadow: 0 0 8px {current_mode["status_color"]};
}}

[data-testid="stChatMessage"] {{
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    backdrop-filter: blur(8px);
}}

[data-testid="stSidebar"] {{
    background: rgba(8, 12, 22, 0.97) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
}}
.sidebar-box {{
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 14px;
    margin-top: 14px;
    margin-bottom: 14px;
}}
.sidebar-box h4 {{
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 10px 0;
}}
.stat-item {{
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 6px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.07);
}}
.stat-item:last-child {{ border-bottom: none; }}
.stat-name {{ color: #64748b; }}
.stat-val {{ color: #cbd5e1; font-weight: 600; }}

.stButton > button {{
    background: rgba(239, 68, 68, 0.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    width: 100%;
}}
.stButton > button:hover {{
    background: rgba(239, 68, 68, 0.35) !important;
    color: #ffffff !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Stats & Clear Button ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-box'>
        <h4>⚙️ Active Mode Info</h4>
        <div class='stat-item'><span class='stat-name'>Mode</span><span class='stat-val'>{selected_mode_name}</span></div>
        <div class='stat-item'><span class='stat-name'>Model</span><span class='stat-val'>gemini-2.5-flash</span></div>
        <div class='stat-item'><span class='stat-name'>Mood</span><span class='stat-val' style='color:{current_mode["accent_color"]};'>{current_mode["status"]}</span></div>
    </div>
    """, unsafe_allow_html=True)

    msg_count = len(st.session_state.messages)
    user_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    bot_count = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    st.markdown(f"""
    <div class='sidebar-box'>
        <h4>📊 Stats</h4>
        <div class='stat-item'><span class='stat-name'>Total Messages</span><span class='stat-val'>{msg_count}</span></div>
        <div class='stat-item'><span class='stat-name'>Your Inputs</span><span class='stat-val'>{user_count}</span></div>
        <div class='stat-item'><span class='stat-name'>Bot Replies</span><span class='stat-val'>{bot_count}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": current_mode["initial_msg"]}
        ]
        st.session_state.langchain_messages = [
            SystemMessage(content=current_mode["system_prompt"]),
            AIMessage(content=current_mode["initial_msg"])
        ]
        st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='header-card'>
    <h1>{current_mode["title"]}</h1>
    <p>{current_mode["tagline"]}</p>
    <div class='status-pill'><span class='status-dot'></span> Mode: {selected_mode_name} • Gemini 2.5 Flash</div>
</div>
""", unsafe_allow_html=True)

# ─── Display Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = current_mode["avatar"] if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ─── Chat Input & Real-time AI Response ───────────────────────────────────────
if prompt := st.chat_input(current_mode["placeholder"]):
    # 1. Add User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Add to LangChain history
    st.session_state.langchain_messages.append(HumanMessage(content=prompt))

    # 3. Stream or Generate Assistant Response
    with st.chat_message("assistant", avatar=current_mode["avatar"]):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner(current_mode["spinner"]):
                response = model.invoke(st.session_state.langchain_messages)
                full_response = response.content
                message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: `{e}`"
            message_placeholder.markdown(full_response)

    # 4. Save Bot message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.langchain_messages.append(AIMessage(content=full_response))


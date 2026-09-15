import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="CineSage — AI Movie Intelligence & Extraction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cinematic CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800;900&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(16, 24, 39, 0.75);
    --bg-card-hover: rgba(24, 35, 58, 0.85);
    --border-color: rgba(255, 215, 0, 0.15);
    --border-glow: rgba(255, 183, 3, 0.35);
    --accent-gold: #ffb703;
    --accent-amber: #fb8500;
    --accent-cyan: #00f0ff;
    --accent-purple: #9d4edd;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

/* Global Font and Body */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background Gradients */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(2, 48, 71, 0.5) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(157, 78, 221, 0.18) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(251, 133, 0, 0.05) 0%, transparent 60%),
                #070a11;
    color: var(--text-main);
}

/* Hero Header */
.hero-container {
    text-align: center;
    padding: 2.2rem 1.5rem 1.8rem 1.5rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(255, 183, 3, 0.05) 0%, rgba(157, 78, 221, 0.08) 100%);
    border: 1px solid rgba(255, 183, 3, 0.2);
    border-radius: 20px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.45), inset 0 0 20px rgba(255, 183, 3, 0.05);
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #ffffff 10%, #ffb703 50%, #fb8500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-transform: uppercase;
    text-shadow: 0 0 30px rgba(255, 183, 3, 0.25);
}

.hero-badge {
    display: inline-block;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    background: rgba(255, 183, 3, 0.12);
    color: var(--accent-gold);
    border: 1px solid rgba(255, 183, 3, 0.3);
    border-radius: 50px;
    margin-bottom: 0.8rem;
}

.hero-subtitle {
    font-size: 1.12rem;
    color: var(--text-muted);
    max-width: 680px;
    margin: 0.5rem auto 0 auto;
    line-height: 1.6;
}

/* Glassmorphic Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}

/* Section Header */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.4rem;
}

/* Meta Pill Badge */
.pill-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 30px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.86rem;
    color: #e2e8f0;
    font-weight: 500;
}

.pill-badge-gold {
    background: rgba(255, 183, 3, 0.12);
    border: 1px solid rgba(255, 183, 3, 0.35);
    color: #ffc93c;
}

.pill-badge-cyan {
    background: rgba(0, 240, 255, 0.1);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #70f4ff;
}

.pill-badge-purple {
    background: rgba(157, 78, 221, 0.12);
    border: 1px solid rgba(157, 78, 221, 0.35);
    color: #c77dff;
}

/* Key Metrics Box */
.metric-tile {
    background: linear-gradient(145deg, rgba(20, 29, 47, 0.8), rgba(11, 16, 26, 0.8));
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.metric-tile-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.metric-tile-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
}

/* Highlight summary banner */
.summary-banner {
    background: linear-gradient(135deg, rgba(251, 133, 0, 0.12) 0%, rgba(157, 78, 221, 0.12) 100%);
    border-left: 4px solid var(--accent-amber);
    padding: 1.2rem 1.4rem;
    border-radius: 0 14px 14px 0;
    margin-bottom: 1.2rem;
    font-size: 1.05rem;
    line-height: 1.65;
    color: #f1f5f9;
}

/* Custom Text Area */
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 0.96rem !important;
    transition: all 0.25s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 15px rgba(255, 183, 3, 0.3) !important;
}

/* Primary Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #ffb703 0%, #fb8500 100%) !important;
    color: #0a0e17 !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 20px rgba(251, 133, 0, 0.35) !important;
    transition: all 0.25s ease !important;
    font-size: 1rem !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(251, 133, 0, 0.55) !important;
    color: #000000 !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Raw code block styling */
pre {
    background: rgba(10, 14, 23, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# Define the EXACT core extraction chain
@st.cache_resource
def get_extraction_chain():
    """Initializes and caches the LangChain extraction model and prompt."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
""" 
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful and important information from the given movie paragraph
and present it in the exact format specified below.

Rules:
- Do NOT add explanations.
- Do NOT add extra commentary.
- Do NOT follow other formats, follow the exact output format.
- Extract ONLY information mentioned in the paragraph.
- Do NOT guess or invent unknown facts.
- If information is missing, write NULL.
- Keep the summary short and clear (2-3 lines maximum).
- Keep the extracted information concise.
- If multiple cast members, genres, themes, or keywords are available,
  list all of them.

Output Format:

Movie Title:
Release Year:
Genre:
Director:
Writer:
Producer:

Cast:
- 

Characters:
- 

Story:
 

Main Conflict:
 

Setting / Location:
 

Themes:
- 

Scientific / Technical Concepts:
- 

Music / Composer:
 

Rating:
 

Rating Source:
 

Reception:
 

Awards / Achievements:
- 

Important Facts:
- 

Keywords:
- 

Quick Summary:
 

Movie Paragraph:
{paragraph}
"""
        ),
        ('human',
"""
Here is the movie paragraph:

{paragraph}
""")
    ])
    return prompt, model


def extract_movie_info(paragraph_text: str) -> str:
    """Executes the extraction chain with identical functionality as core.py."""
    prompt, model = get_extraction_chain()
    final_prompt = prompt.invoke({"paragraph": paragraph_text})
    response = model.invoke(final_prompt)
    return response.content


def parse_extracted_data(text: str) -> dict:
    """Parses formatted extraction output into structured key-value pairs for rich UI rendering."""
    data = {}
    
    # List of known scalar and list keys
    keys = [
        "Movie Title", "Release Year", "Genre", "Director", "Writer", "Producer",
        "Cast", "Characters", "Story", "Main Conflict", "Setting / Location",
        "Themes", "Scientific / Technical Concepts", "Music / Composer",
        "Rating", "Rating Source", "Reception", "Awards / Achievements",
        "Important Facts", "Keywords", "Quick Summary"
    ]
    
    # Regex pattern to match each section header
    pattern = r'(?:^|\n)(' + '|'.join([re.escape(k) for k in keys]) + r'):\s*(.*?)(?=\n(?:' + '|'.join([re.escape(k) for k in keys]) + r'):|$)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    for key, val in matches:
        clean_val = val.strip()
        # Handle list format
        if key in ["Cast", "Characters", "Themes", "Scientific / Technical Concepts", "Awards / Achievements", "Important Facts", "Keywords"]:
            items = [re.sub(r'^-\s*', '', line).strip() for line in clean_val.split('\n') if line.strip() and line.strip() != '-']
            # Filter out NULL item list
            if len(items) == 1 and items[0].upper() == "NULL":
                items = []
            data[key] = items if items else (["NULL"] if clean_val.upper() == "NULL" else [clean_val])
        else:
            data[key] = clean_val if clean_val else "NULL"
            
    return data


# Preset sample paragraphs for instant testing
SAMPLE_MOVIES = {
    "Interstellar (2014)": (
        "Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. "
        "Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, "
        "and Michael Caine. The story revolves around a group of astronauts who travel through a "
        "wormhole near Saturn in search of a new home for humanity as Earth faces environmental "
        "collapse. The movie was widely appreciated for its emotional depth, scientific accuracy, "
        "and Hans Zimmer’s powerful soundtrack. It holds a rating of 8.6 on IMDb and is often "
        "considered one of the greatest sci-fi films of the 21st century."
    ),
    "Inception (2010)": (
        "Inception is a 2010 mind-bending science fiction action film written and directed by Christopher Nolan. "
        "The film stars Leonardo DiCaprio as Dom Cobb, a professional thief who extracts information from the subconscious "
        "of his targets during dream states. Co-starring Joseph Gordon-Levitt, Elliot Page, and Tom Hardy, the plot follows "
        "Cobb being offered a chance to have his criminal history erased as payment for the implantation of another person's idea "
        "into a target's subconscious. The film received praise for its original screenplay, visual effects, and Hans Zimmer's score, "
        "winning four Academy Awards and scoring an 8.8 on IMDb."
    ),
    "Oppenheimer (2023)": (
        "Oppenheimer is a 2023 biographical drama written and directed by Christopher Nolan. Based on the 2005 biography "
        "American Prometheus, the film chronicles the life of J. Robert Oppenheimer, the theoretical physicist who led the Manhattan Project "
        "in Los Alamos to develop the first nuclear weapons. Cillian Murphy stars in the titular role alongside Emily Blunt, Matt Damon, "
        "and Robert Downey Jr. The film grossed over $957 million worldwide and won seven Academy Awards including Best Picture and Best Director."
    ),
    "The Dark Knight (2008)": (
        "The Dark Knight is a 2008 superhero film directed by Christopher Nolan and co-written with Jonathan Nolan. "
        "Christian Bale stars as Bruce Wayne / Batman alongside Heath Ledger as the psychopathic criminal mastermind known as the Joker. "
        "Set in Gotham City, Batman forms an alliance with police lieutenant James Gordon and district attorney Harvey Dent to dismantle organized crime. "
        "Heath Ledger's posthumous performance received universal acclaim, earning him the Academy Award for Best Supporting Actor. "
        "It holds a 9.0 on IMDb and redefined comic book cinema."
    )
}

# --- Sidebar Configuration & Info ---
with st.sidebar:
    st.markdown("### 🎬 CineSage Control Panel")
    st.caption("Powered by Google Gemini 2.5 Flash & LangChain")
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Presets")
    selected_sample = st.selectbox(
        "Choose a sample movie paragraph:",
        ["-- Select a Sample --"] + list(SAMPLE_MOVIES.keys())
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Details")
    st.markdown("""
    - **Engine**: `gemini-2.5-flash`
    - **Extraction Mode**: Zero-shot structured parser
    - **Strict Rules**: No hallucination, `NULL` fallback
    """)
    
    st.markdown("---")
    st.markdown("#### 📜 Extraction History")
    if "history" not in st.session_state:
        st.session_state.history = []
        
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**{item.get('title', 'Untitled')}** ({item.get('year', 'N/A')})")
            st.caption(f"Genre: {item.get('genre', 'N/A')}")
            st.markdown("---")
    else:
        st.caption("No extractions in this session yet.")
        
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --- Main Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-Powered Cinematic Entity Extraction</div>
    <h1 class="hero-title">CINESAGE</h1>
    <p class="hero-subtitle">
        Transform unstructured movie synopses, reviews, and paragraphs into high-fidelity structured intelligence cards in milliseconds.
    </p>
</div>
""", unsafe_allow_html=True)


# --- Input Section ---
default_text = ""
if selected_sample != "-- Select a Sample --":
    default_text = SAMPLE_MOVIES[selected_sample]

col_left, col_right = st.columns([2.8, 1.2])

with col_left:
    user_input = st.text_area(
        "Movie Paragraph / Synopsis Text:",
        value=default_text,
        height=180,
        placeholder="Paste your movie paragraph, plot synopsis, or review here...",
        help="Input any paragraph describing a movie. CineSage will extract title, cast, conflict, themes, technical concepts, and more."
    )

with col_right:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    extract_clicked = st.button("🚀 Extract Movie Intelligence", use_container_width=True)
    
    if st.button("🧹 Clear Input", use_container_width=True):
        st.rerun()
        
    st.markdown("""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; margin-top: 12px;'>
        <div style='font-size: 0.8rem; color: #94a3b8;'>💡 <b>Tip:</b> Include plot points, director, cast, sound design, and ratings in your paragraph for comprehensive extraction.</div>
    </div>
    """, unsafe_allow_html=True)


# --- Extraction Execution ---
if extract_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please provide a movie paragraph or select one of the presets before extracting.")
    else:
        with st.spinner("Analyzing cinematic entities and generating structured profile..."):
            try:
                raw_result = extract_movie_info(user_input.strip())
                parsed_data = parse_extracted_data(raw_result)
                
                # Save to session history
                movie_title = parsed_data.get("Movie Title", "Untitled")
                release_year = parsed_data.get("Release Year", "N/A")
                genre = parsed_data.get("Genre", "N/A")
                
                st.session_state.last_extraction = {
                    "raw": raw_result,
                    "parsed": parsed_data,
                    "input": user_input.strip()
                }
                st.session_state.history.append({
                    "title": movie_title,
                    "year": release_year,
                    "genre": genre
                })
            except Exception as e:
                st.error(f"❌ An error occurred during extraction: {str(e)}")


# --- Display Results ---
if "last_extraction" in st.session_state:
    data = st.session_state.last_extraction["parsed"]
    raw_text = st.session_state.last_extraction["raw"]
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    # Quick Executive Summary Highlight
    quick_summary = data.get("Quick Summary", "")
    if quick_summary and quick_summary.upper() != "NULL":
        st.markdown(f"""
        <div class="summary-banner">
            <b style="color: var(--accent-gold); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px;">📌 Quick Summary</b>
            {quick_summary}
        </div>
        """, unsafe_allow_html=True)
    
    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Movie Title</div>
            <div class="metric-tile-value" style="color: var(--accent-gold);">{data.get("Movie Title", "NULL")}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Release Year</div>
            <div class="metric-tile-value">{data.get("Release Year", "NULL")}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Genre</div>
            <div class="metric-tile-value" style="color: var(--accent-cyan); font-size: 1.1rem;">{data.get("Genre", "NULL")}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Rating</div>
            <div class="metric-tile-value" style="color: #4ade80;">⭐ {data.get("Rating", "NULL")}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col5:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Director</div>
            <div class="metric-tile-value" style="font-size: 1.1rem;">{data.get("Director", "NULL")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Tabs for Visual Dossier vs Exact Raw Output vs Export
    tab_dossier, tab_raw, tab_export = st.tabs(["✨ Visual Intelligence Dossier", "📋 Exact Raw Output", "💾 Export & Raw JSON"])
    
    with tab_dossier:
        col_main1, col_main2 = st.columns([1.2, 1])
        
        with col_main1:
            # Narrative & Plot Overview
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">📖 Narrative & Storyline</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Plot Synopsis:**\n{data.get('Story', 'NULL')}")
            
            main_conflict = data.get("Main Conflict", "")
            if main_conflict and main_conflict.upper() != "NULL":
                st.markdown(f"<div style='margin-top: 10px;'><b>⚡ Main Conflict:</b><br><span style='color: #cbd5e1;'>{main_conflict}</span></div>", unsafe_allow_html=True)
                
            setting = data.get("Setting / Location", "")
            if setting and setting.upper() != "NULL":
                st.markdown(f"<div style='margin-top: 10px;'><b>📍 Setting / Location:</b><br><span style='color: #cbd5e1;'>{setting}</span></div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Cast & Characters
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">🎭 Cast & Key Characters</div>
            """, unsafe_allow_html=True)
            
            cast_list = data.get("Cast", [])
            char_list = data.get("Characters", [])
            
            st.markdown("<b>Starring Cast:</b>", unsafe_allow_html=True)
            if cast_list and cast_list != ["NULL"]:
                badges = " ".join([f"<span class='pill-badge pill-badge-gold'>👤 {actor}</span>" for actor in cast_list])
                st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top: 12px;'><b>Characters Identified:</b></div>", unsafe_allow_html=True)
            if char_list and char_list != ["NULL"]:
                char_badges = " ".join([f"<span class='pill-badge pill-badge-cyan'>🎭 {c}</span>" for c in char_list])
                st.markdown(f"<div>{char_badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

            # Themes & Scientific Concepts
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">🧬 Core Themes & Technical Concepts</div>
            """, unsafe_allow_html=True)
            
            themes = data.get("Themes", [])
            concepts = data.get("Scientific / Technical Concepts", [])
            
            st.markdown("<b>Themes:</b>", unsafe_allow_html=True)
            if themes and themes != ["NULL"]:
                theme_badges = " ".join([f"<span class='pill-badge pill-badge-purple'>💡 {t}</span>" for t in themes])
                st.markdown(f"<div>{theme_badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top: 12px;'><b>Scientific / Technical Concepts:</b></div>", unsafe_allow_html=True)
            if concepts and concepts != ["NULL"]:
                concept_badges = " ".join([f"<span class='pill-badge pill-badge-cyan'>🔬 {concept}</span>" for concept in concepts])
                st.markdown(f"<div>{concept_badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_main2:
            # Production & Creative Team
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">🎬 Production & Crew</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"• **Director:** `{data.get('Director', 'NULL')}`")
            st.markdown(f"• **Writer:** `{data.get('Writer', 'NULL')}`")
            st.markdown(f"• **Producer:** `{data.get('Producer', 'NULL')}`")
            st.markdown(f"• **Music / Composer:** `{data.get('Music / Composer', 'NULL')}`")
            st.markdown("</div>", unsafe_allow_html=True)

            # Critical Reception & Ratings
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">🏆 Critical Reception & Accolades</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"• **Rating:** `{data.get('Rating', 'NULL')}` (Source: `{data.get('Rating Source', 'NULL')}`)")
            st.markdown(f"• **Reception:** {data.get('Reception', 'NULL')}")
            
            awards = data.get("Awards / Achievements", [])
            st.markdown("<div style='margin-top: 10px;'><b>Awards & Accolades:</b></div>", unsafe_allow_html=True)
            if awards and awards != ["NULL"]:
                for aw in awards:
                    st.markdown(f"- 🏅 {aw}")
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

            # Important Facts & Keywords
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">🏷️ Key Tags & Facts</div>
            """, unsafe_allow_html=True)
            
            keywords = data.get("Keywords", [])
            if keywords and keywords != ["NULL"]:
                kw_badges = " ".join([f"<span class='pill-badge'>#{kw}</span>" for kw in keywords])
                st.markdown(f"<div>{kw_badges}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            facts = data.get("Important Facts", [])
            st.markdown("<div style='margin-top: 12px;'><b>Important Facts:</b></div>", unsafe_allow_html=True)
            if facts and facts != ["NULL"]:
                for f in facts:
                    st.markdown(f"- 📌 {f}")
            else:
                st.markdown("<span style='color: #64748b;'>NULL</span>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
    with tab_raw:
        st.markdown("#### Output Generated by Model (Exact Format)")
        st.code(raw_text, language="yaml")
        st.download_button(
            label="📥 Download Raw Text Output",
            data=raw_text,
            file_name=f"{data.get('Movie Title', 'movie').replace(' ', '_').lower()}_extracted.txt",
            mime="text/plain"
        )
        
    with tab_export:
        st.markdown("#### Export Structured JSON")
        json_output = json.dumps(data, indent=2)
        st.code(json_output, language="json")
        st.download_button(
            label="📥 Download Structured JSON",
            data=json_output,
            file_name=f"{data.get('Movie Title', 'movie').replace(' ', '_').lower()}_data.json",
            mime="application/json"
        )

# Footer
st.markdown("""
<div style='text-align: center; color: #475569; font-size: 0.85rem; margin-top: 3.5rem; padding-bottom: 2rem;'>
    CineSage • Advanced Entity Extraction Engine • Built with LangChain & Streamlit
</div>
""", unsafe_allow_html=True)

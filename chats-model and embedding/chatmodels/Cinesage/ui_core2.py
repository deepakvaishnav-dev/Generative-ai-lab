import os
import json
import time
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CineSage 2.0 — AI Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cinematic Cyber-Glassmorphism CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #07090e;
    --bg-card: rgba(13, 19, 33, 0.78);
    --bg-card-hover: rgba(20, 29, 49, 0.9);
    --border-subtle: rgba(255, 215, 0, 0.12);
    --border-glow: rgba(255, 183, 3, 0.45);
    --accent-gold: #ffb703;
    --accent-amber: #fb8500;
    --accent-cyan: #00f0ff;
    --accent-purple: #a855f7;
    --accent-emerald: #10b981;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

/* Global Typography */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Main Background */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(2, 48, 71, 0.55) 0%, transparent 40%),
                radial-gradient(circle at 90% 85%, rgba(168, 85, 247, 0.2) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(251, 133, 0, 0.04) 0%, transparent 60%),
                #06080d;
    color: var(--text-primary);
}

/* Hero Banner */
.hero-header {
    text-align: center;
    padding: 2.2rem 1.8rem 1.8rem 1.8rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(135deg, rgba(255, 183, 3, 0.06) 0%, rgba(168, 85, 247, 0.09) 100%);
    border: 1px solid rgba(255, 183, 3, 0.22);
    border-radius: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    background: rgba(255, 183, 3, 0.12);
    color: var(--accent-gold);
    border: 1px solid rgba(255, 183, 3, 0.35);
    border-radius: 100px;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3.1rem;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #ffffff 15%, #ffd166 50%, #fb8500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-transform: uppercase;
    text-shadow: 0 0 35px rgba(255, 183, 3, 0.2);
}

.hero-subtitle {
    font-size: 1.08rem;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0.6rem auto 0 auto;
    line-height: 1.6;
}

/* Glass Movie Card Container */
.movie-showcase-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.45);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.movie-showcase-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-gold), var(--accent-purple), var(--accent-cyan));
}

.movie-title-display {
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
    letter-spacing: 1px;
}

/* Badges & Chips */
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 30px;
    font-size: 0.84rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
    letter-spacing: 0.5px;
}

.badge-gold {
    background: rgba(255, 183, 3, 0.12);
    border: 1px solid rgba(255, 183, 3, 0.35);
    color: #ffd166;
}

.badge-cyan {
    background: rgba(0, 240, 255, 0.1);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #70f4ff;
}

.badge-purple {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.35);
    color: #d8b4fe;
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
}

/* Metric Pill Box */
.stat-box {
    background: rgba(10, 15, 26, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}

.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
    margin-bottom: 2px;
}

.stat-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
}

/* Summary Box */
.summary-container {
    background: linear-gradient(135deg, rgba(251, 133, 0, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
    border-left: 4px solid var(--accent-gold);
    padding: 1.2rem 1.4rem;
    border-radius: 0 14px 14px 0;
    margin: 1.2rem 0;
    font-size: 1.02rem;
    line-height: 1.65;
    color: #f1f5f9;
}

/* Interactive Input Controls */
.stTextArea textarea {
    background-color: rgba(13, 19, 33, 0.85) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
    font-size: 0.98rem !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 18px rgba(255, 183, 3, 0.25) !important;
}

/* Primary Extract Button */
.stButton > button {
    background: linear-gradient(135deg, #ffb703 0%, #fb8500 100%) !important;
    color: #06080d !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.8px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.8rem !important;
    box-shadow: 0 4px 20px rgba(251, 133, 0, 0.35) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(251, 133, 0, 0.55) !important;
    color: #000000 !important;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #07090e !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Code Display */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. CORE FUNCTIONALITY & PYDANTIC SCHEMA
# ==========================================
class MovieInfo(BaseModel):
    title: str = Field(description="The primary title of the movie")
    release_year: Optional[int] = Field(None, description="The year the movie was released")
    genre: List[str] = Field(default_factory=list, description="List of movie genres")
    director: Optional[str] = Field(None, description="Director of the movie")
    cast: List[str] = Field(default_factory=list, description="List of leading actors/cast members")
    rating: Optional[float] = Field(None, description="Movie rating score (e.g. IMDb or Rotten Tomatoes scaled out of 10)")
    summary: str = Field(description="A concise summary of the movie's plot and significance")


def get_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=MovieInfo)


def get_extraction_chain(api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
    """Initializes the LangChain extraction prompt and model."""
    kwargs = {"model": model_name, "temperature": temperature}
    if api_key:
        kwargs["google_api_key"] = api_key
        
    model = ChatGoogleGenerativeAI(**kwargs)
    parser = get_parser()
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', """Extract movie information from the paragraph.
{format_instructions}
"""),
        ("human", "{paragraph}")
    ])
    
    return prompt, model, parser


def execute_extraction(paragraph_text: str, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
    """Executes the extraction chain with identical core logic as core.py."""
    prompt, model, parser = get_extraction_chain(api_key=api_key, model_name=model_name, temperature=temperature)
    
    final_prompt = prompt.invoke({
        "paragraph": paragraph_text,
        "format_instructions": parser.get_format_instructions()
    })
    
    response = model.invoke(final_prompt)
    parsed_obj: MovieInfo = parser.invoke(response)
    return parsed_obj, response.content


# ==========================================
# 2. SAMPLE PRESET DATA
# ==========================================
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
        "into a target's subconscious. The film received universal acclaim, winning four Academy Awards and scoring an 8.8 on IMDb."
    ),
    "Oppenheimer (2023)": (
        "Oppenheimer is a 2023 biographical drama written and directed by Christopher Nolan. Based on the 2005 biography "
        "American Prometheus, the film chronicles the life of J. Robert Oppenheimer, the theoretical physicist who led the Manhattan Project "
        "in Los Alamos to develop the first nuclear weapons. Cillian Murphy stars in the titular role alongside Emily Blunt, Matt Damon, "
        "and Robert Downey Jr. The film grossed over $957 million worldwide, won seven Academy Awards including Best Picture, and holds an 8.9 rating."
    ),
    "The Dark Knight (2008)": (
        "The Dark Knight is a 2008 superhero film directed by Christopher Nolan and co-written with Jonathan Nolan. "
        "Christian Bale stars as Bruce Wayne / Batman alongside Heath Ledger as the psychopathic criminal mastermind known as the Joker. "
        "Set in Gotham City, Batman forms an alliance with police lieutenant James Gordon and district attorney Harvey Dent to dismantle organized crime. "
        "Heath Ledger's posthumous performance received universal acclaim, earning him the Academy Award for Best Supporting Actor. "
        "It holds a 9.0 on IMDb and redefined modern comic book cinema."
    ),
    "Dune: Part Two (2024)": (
        "Dune: Part Two is a 2024 epic science fiction film directed by Denis Villeneuve. The film stars Timothée Chalamet as Paul Atreides, "
        "Zendaya as Chani, Rebecca Ferguson, Javier Bardem, and Austin Butler. The story follows Paul Atreides as he unites with the Fremen people "
        "of the desert planet Arrakis to wage war against House Harkonnen and avenge his fallen family. Celebrated for its groundbreaking cinematography, "
        "world-building, and sound design, the film holds an 8.5 rating on IMDb."
    )
}


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "current_text" not in st.session_state:
    st.session_state.current_text = SAMPLE_MOVIES["Interstellar (2014)"]

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "raw_llm_response" not in st.session_state:
    st.session_state.raw_llm_response = None


# ==========================================
# 4. SIDEBAR CONFIGURATION & HISTORY
# ==========================================
with st.sidebar:
    st.markdown("### 🎬 CineSage 2.0")
    st.caption("AI-Powered Structured Movie Extraction")
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Presets")
    
    for title, text in SAMPLE_MOVIES.items():
        if st.button(f"📌 {title}", use_container_width=True, key=f"btn_{title}"):
            st.session_state.current_text = text
            st.rerun()
            
    st.markdown("---")
    st.markdown("#### ⚙️ LLM Settings")
    selected_model = st.selectbox(
        "Model Engine",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", placeholder="Leave blank to use .env")
    api_key = api_key_input.strip() if api_key_input.strip() else None

    st.markdown("---")
    st.markdown("#### 📜 Extraction History")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-6:])):
            with st.container():
                st.markdown(f"**{item.title}** ({item.release_year or 'N/A'})")
                st.caption(f"⭐ {item.rating or 'N/A'} | {', '.join(item.genre[:2])}")
                st.markdown("---")
    else:
        st.caption("No extraction history yet.")


# ==========================================
# 5. HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✨ Pydantic Structured Output 2.0</div>
    <h1 class="hero-title">CineSage Intelligence</h1>
    <p class="hero-subtitle">
        Extract comprehensive, strictly-validated movie metadata, cast, director, ratings, and narrative summaries from any unstructured text or review.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 6. INPUT INTERFACE
# ==========================================
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    st.markdown("#### 📝 Movie Paragraph or Review")
with col_in2:
    word_count = len(st.session_state.current_text.split()) if st.session_state.current_text else 0
    char_count = len(st.session_state.current_text) if st.session_state.current_text else 0
    st.markdown(f"<p style='text-align:right; color: #94a3b8; font-size: 0.85rem; margin-top: 5px;'>{word_count} words | {char_count} chars</p>", unsafe_allow_html=True)

input_text = st.text_area(
    label="Movie Input Paragraph",
    value=st.session_state.current_text,
    height=160,
    placeholder="Paste any movie synopsis, plot review, Wikipedia excerpt, or paragraph here...",
    label_visibility="collapsed"
)

# Update session text if user edits
st.session_state.current_text = input_text

col_btn1, col_btn2 = st.columns([4, 1])
with col_btn1:
    extract_clicked = st.button("🚀 Extract Movie Intelligence", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.current_text = ""
        st.session_state.last_result = None
        st.session_state.raw_llm_response = None
        st.rerun()


# ==========================================
# 7. PROCESSING & EXTRACTION PIPELINE
# ==========================================
if extract_clicked:
    if not input_text.strip():
        st.warning("⚠️ Please provide a movie paragraph or select one of the presets above.")
    else:
        with st.spinner("🎬 Analyzing narrative with Gemini & validating Pydantic schema..."):
            try:
                start_time = time.time()
                parsed_data, raw_resp = execute_extraction(
                    paragraph_text=input_text,
                    api_key=api_key,
                    model_name=selected_model,
                    temperature=temperature
                )
                elapsed = time.time() - start_time
                
                st.session_state.last_result = parsed_data
                st.session_state.raw_llm_response = raw_resp
                
                # Append to history if not duplicate of latest
                if not st.session_state.history or st.session_state.history[-1].title != parsed_data.title:
                    st.session_state.history.append(parsed_data)
                    
                st.success(f"✨ Successfully extracted and validated in {elapsed:.2f}s!")
            except Exception as e:
                st.error(f"❌ Extraction Error: {str(e)}")


# ==========================================
# 8. VISUAL MOVIE SHOWCASE CARD
# ==========================================
if st.session_state.last_result:
    movie: MovieInfo = st.session_state.last_result
    
    st.markdown("---")
    st.markdown("### 🎬 Extracted Movie Showcase")
    
    # Rating display formatting
    rating_display = f"⭐ {movie.rating:.1f} / 10" if movie.rating else "N/A"
    year_display = str(movie.release_year) if movie.release_year else "Unknown Year"
    director_display = movie.director if movie.director else "Unknown Director"
    
    # Showcase Hero Card
    st.markdown(f"""
    <div class="movie-showcase-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <span class="badge-chip badge-gold">FILM DOSSIER</span>
                <span class="badge-chip badge-purple">📅 {year_display}</span>
                <span class="badge-chip badge-cyan">🎬 Directed by {director_display}</span>
                <h1 class="movie-title-display">{movie.title}</h1>
            </div>
            <div style="text-align: right;">
                <span class="badge-chip badge-emerald" style="font-size: 1.15rem; padding: 8px 16px;">{rating_display}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Metadata Columns
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    with col_m1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Release Year</div>
            <div class="stat-value">{}</div>
        </div>
        """.format(year_display), unsafe_allow_html=True)
        
    with col_m2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Director</div>
            <div class="stat-value" style="font-size: 1.1rem;">{}</div>
        </div>
        """.format(director_display), unsafe_allow_html=True)
        
    with col_m3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Audience Rating</div>
            <div class="stat-value" style="color: var(--accent-gold);">{}</div>
        </div>
        """.format(rating_display), unsafe_allow_html=True)
        
    # Genres & Cast Badges
    st.markdown("<br>", unsafe_allow_html=True)
    col_tag1, col_tag2 = st.columns([1, 1])
    
    with col_tag1:
        st.markdown("##### 🎭 Genres")
        if movie.genre:
            genre_html = "".join([f'<span class="badge-chip badge-purple">{g.strip()}</span>' for g in movie.genre])
            st.markdown(genre_html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #64748b;'>No genres detected</span>", unsafe_allow_html=True)
            
    with col_tag2:
        st.markdown("##### 🌟 Starring Cast")
        if movie.cast:
            cast_html = "".join([f'<span class="badge-chip badge-gold">👤 {c.strip()}</span>' for c in movie.cast])
            st.markdown(cast_html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #64748b;'>No cast listed</span>", unsafe_allow_html=True)
            
    # Narrative Summary Callout
    st.markdown("##### 📖 Narrative Summary")
    st.markdown(f"""
    <div class="summary-container">
        {movie.summary}
    </div>
    """, unsafe_allow_html=True)
    
    # Export & Code Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab_json, tab_md, tab_raw = st.tabs(["📦 JSON Object", "📄 Markdown Report", "🔍 Raw Model Output"])
    
    movie_dict = movie.model_dump()
    json_string = json.dumps(movie_dict, indent=2)
    
    with tab_json:
        st.json(movie_dict)
        st.download_button(
            label="💾 Download JSON Data",
            data=json_string,
            file_name=f"{movie.title.lower().replace(' ', '_')}_info.json",
            mime="application/json"
        )
        
    with tab_md:
        md_content = f"""# {movie.title} ({movie.release_year or 'N/A'})

**Director:** {movie.director or 'N/A'}  
**Rating:** {movie.rating or 'N/A'}/10  
**Genres:** {', '.join(movie.genre)}  
**Cast:** {', '.join(movie.cast)}  

### Synopsis
{movie.summary}
"""
        st.code(md_content, language="markdown")
        st.download_button(
            label="💾 Download Markdown Report",
            data=md_content,
            file_name=f"{movie.title.lower().replace(' ', '_')}_report.md",
            mime="text/markdown"
        )
        
    with tab_raw:
        st.markdown("##### Direct LLM Output Before Pydantic Validation:")
        st.code(st.session_state.raw_llm_response or "No raw response cached.", language="json")

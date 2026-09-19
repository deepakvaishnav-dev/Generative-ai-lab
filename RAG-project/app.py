import os
import sys
import json
import uuid
import tempfile
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Storage Configuration for Persistent Chat History ---
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


def load_persistent_history():
    """Load all chat sessions from disk."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "sessions" in data:
                    return data
        except Exception:
            pass
    # Default structure if file does not exist or is corrupted
    return {"sessions": {}, "current_session_id": None}


def save_persistent_history(data):
    """Save all chat sessions to disk."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


def get_new_session_id():
    """Generate a unique session ID."""
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


# --- Page Configuration ---
st.set_page_config(
    page_title="DocuMind RAG | Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Premium CSS ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1050px;
    }

    /* Header Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 6px;
    }
    .badge-active {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-info {
        background-color: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    /* Source Box / Chunk Cards */
    .source-chunk {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #cbd5e1;
    }
    .source-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }

    /* Chat Session List in Sidebar */
    .session-item {
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s;
    }
    .session-item-active {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* Sidebar Clean styling */
    section[data-testid="stSidebar"] {
        background: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
</style>
""", unsafe_allow_html=True)


# --- Cached Resources ---
@st.cache_resource(show_spinner="Loading Embedding Model (sentence-transformers/all-mpnet-base-v2)...")
def get_embedding_model():
    """Cache the HuggingFace embedding model so it only loads once in memory."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def build_vectorstore_from_pdf(uploaded_file, embedding_model):
    """Processes uploaded PDF file into text chunks and Chroma vectorstore."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # In-memory session vectorstore for the uploaded document
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
        return vectorstore, len(docs), len(chunks)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_existing_chromadb(embedding_model):
    """Loads existing Chroma database from disk."""
    if not os.path.exists("chroma-db"):
        return None
    try:
        vectorstore = Chroma(
            persist_directory="chroma-db",
            embedding_function=embedding_model,
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading existing Chroma DB: {e}")
        return None


# --- Initialize Persistent Chat History & Sessions in Session State ---
persistent_data = load_persistent_history()

if "sessions" not in st.session_state:
    st.session_state.sessions = persistent_data.get("sessions", {})

if "current_session_id" not in st.session_state:
    saved_cur_id = persistent_data.get("current_session_id")
    if saved_cur_id and saved_cur_id in st.session_state.sessions:
        st.session_state.current_session_id = saved_cur_id
    elif st.session_state.sessions:
        # Pick the most recent session
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
    else:
        # Create initial empty session
        new_id = get_new_session_id()
        st.session_state.sessions[new_id] = {
            "id": new_id,
            "title": "New Chat",
            "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            "messages": []
        }
        st.session_state.current_session_id = new_id
        save_persistent_history({
            "sessions": st.session_state.sessions,
            "current_session_id": new_id
        })

# Ensure current session exists
if st.session_state.current_session_id not in st.session_state.sessions:
    new_id = get_new_session_id()
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": "New Chat",
        "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "messages": []
    }
    st.session_state.current_session_id = new_id
    save_persistent_history({
        "sessions": st.session_state.sessions,
        "current_session_id": new_id
    })

# Other runtime session variables
if "custom_vectorstore" not in st.session_state:
    st.session_state.custom_vectorstore = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

embedding_model = get_embedding_model()

# --- Sidebar UI ---
with st.sidebar:
    # ➕ New Chat Action Button
    st.markdown("### 💬 Conversations")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = get_new_session_id()
        st.session_state.sessions[new_id] = {
            "id": new_id,
            "title": "New Chat",
            "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            "messages": []
        }
        st.session_state.current_session_id = new_id
        save_persistent_history({
            "sessions": st.session_state.sessions,
            "current_session_id": new_id
        })
        st.rerun()

    # Chat History List (Stored History)
    with st.expander(f"📜 Chat History ({len(st.session_state.sessions)})", expanded=True):
        if not st.session_state.sessions:
            st.caption("No conversations saved yet.")
        else:
            # Sort sessions by creation/recency
            session_ids = list(st.session_state.sessions.keys())
            session_ids.reverse()  # Newest first

            for sid in session_ids:
                s_data = st.session_state.sessions[sid]
                is_active = (sid == st.session_state.current_session_id)
                s_title = s_data.get("title", "Untitled Chat")
                s_date = s_data.get("created_at", "")
                msg_count = len(s_data.get("messages", []))

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    # Highlight active chat with icon
                    icon = "👉 " if is_active else "🗨️ "
                    btn_label = f"{icon}{s_title[:22]}... ({msg_count})" if len(s_title) > 22 else f"{icon}{s_title} ({msg_count})"
                    if st.button(
                        btn_label,
                        key=f"sess_{sid}",
                        use_container_width=True,
                        help=f"Created: {s_date}\nClick to switch to this chat."
                    ):
                        st.session_state.current_session_id = sid
                        save_persistent_history({
                            "sessions": st.session_state.sessions,
                            "current_session_id": sid
                        })
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_{sid}", help="Delete this chat"):
                        # Delete this session
                        st.session_state.sessions.pop(sid, None)
                        if st.session_state.current_session_id == sid:
                            if st.session_state.sessions:
                                st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]
                            else:
                                fresh_id = get_new_session_id()
                                st.session_state.sessions[fresh_id] = {
                                    "id": fresh_id,
                                    "title": "New Chat",
                                    "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                                    "messages": []
                                }
                                st.session_state.current_session_id = fresh_id

                        save_persistent_history({
                            "sessions": st.session_state.sessions,
                            "current_session_id": st.session_state.current_session_id
                        })
                        st.rerun()

        # Clear All Chats Button with confirmation
        st.markdown("---")
        if st.button("⚠️ Clear All Chat History", use_container_width=True):
            fresh_id = get_new_session_id()
            st.session_state.sessions = {
                fresh_id: {
                    "id": fresh_id,
                    "title": "New Chat",
                    "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                    "messages": []
                }
            }
            st.session_state.current_session_id = fresh_id
            save_persistent_history({
                "sessions": st.session_state.sessions,
                "current_session_id": fresh_id
            })
            st.rerun()

    st.divider()

    # Knowledge Base Selection
    st.markdown("### 📚 Knowledge Base")
    data_source_mode = st.radio(
        "Select Document Source:",
        ["📄 Upload New PDF", "🗄️ Existing Database (chroma-db)"],
        index=0
    )

    active_vectorstore = None

    if data_source_mode == "📄 Upload New PDF":
        st.markdown("#### Upload PDF File")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file to index",
            type=["pdf"],
            help="Upload any PDF to query it using RAG"
        )

        if uploaded_pdf is not None:
            # Check if this is a newly uploaded file
            if (st.session_state.doc_stats is None or 
                st.session_state.doc_stats.get("name") != uploaded_pdf.name):
                
                with st.spinner(f"Processing and embedding '{uploaded_pdf.name}'..."):
                    try:
                        vstore, total_pages, total_chunks = build_vectorstore_from_pdf(uploaded_pdf, embedding_model)
                        st.session_state.custom_vectorstore = vstore
                        st.session_state.doc_stats = {
                            "name": uploaded_pdf.name,
                            "pages": total_pages,
                            "chunks": total_chunks,
                            "size_kb": round(len(uploaded_pdf.getvalue()) / 1024, 1)
                        }
                        st.success("✅ Document indexed successfully!")
                    except Exception as err:
                        st.error(f"Failed to process PDF: {err}")

            if st.session_state.custom_vectorstore is not None:
                active_vectorstore = st.session_state.custom_vectorstore
                stats = st.session_state.doc_stats
                st.markdown(f"""
                <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 12px; margin-top: 10px;">
                    <div style="font-weight: 600; color: #38bdf8; font-size: 0.9rem;">📄 {stats['name']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                        • Pages: <b>{stats['pages']}</b><br>
                        • Text Chunks: <b>{stats['chunks']}</b><br>
                        • File Size: <b>{stats['size_kb']} KB</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Upload a PDF above to start questioning your document.")
            st.session_state.custom_vectorstore = None
            st.session_state.doc_stats = None

    else:
        # Existing Chroma DB Mode
        with st.spinner("Connecting to existing chroma-db..."):
            existing_store = load_existing_chromadb(embedding_model)
            if existing_store:
                active_vectorstore = existing_store
                st.success("✅ Connected to `chroma-db` collection")
                st.caption("Using pre-indexed vectorstore created from `create_database.py`.")
            else:
                st.warning("⚠️ No existing `chroma-db` found. Please run `create_database.py` or switch to PDF Upload mode.")

    st.divider()

    # Model & Retrieval Hyperparameters
    st.markdown("### 🤖 LLM & Retriever Settings")
    
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=[
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0,
        help="Select the model hosted on Groq"
    )

    search_type = st.selectbox(
        "Search Strategy",
        options=["mmr", "similarity"],
        index=0,
        help="MMR (Maximal Marginal Relevance) reduces redundancy in retrieved chunks"
    )

    top_k = st.slider(
        "Number of Chunks (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="How many document chunks to feed as context"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values are more factual and strict"
    )

# --- Active Session Messages Reference ---
current_session = st.session_state.sessions[st.session_state.current_session_id]
current_messages = current_session.setdefault("messages", [])

# --- Main App Header ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🧠 DocuMind RAG Assistant</div>
        <p class="hero-subtitle">Intelligent Document Q&A with Persistent Chat History, LangChain, ChromaDB & Groq</p>
        <div style="margin-top: 12px;">
            <span class="badge badge-active">Embeddings: all-mpnet-base-v2</span>
            <span class="badge badge-info">Search: {} (k={})</span>
        </div>
    </div>
    """.format(search_type.upper(), top_k), unsafe_allow_html=True)

with col_head2:
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; margin-top: 8px;">
        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Current Session</div>
        <div style="font-size: 1rem; color: #f1f5f9; font-weight: 600; margin-top: 2px;">💬 {current_session.get('title', 'New Chat')}</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">{current_session.get('created_at', '')} • {len(current_messages)} messages</div>
    </div>
    """, unsafe_allow_html=True)

# --- Check Groq API Key ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not detected in `.env`. Please add it to your `.env` file to chat with the LLM.")

# --- Render Current Session's Chat History ---
for msg in current_messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(content)
        if sources:
            with st.expander(f"🔍 View {len(sources)} Retrieved Context Chunks", expanded=False):
                for idx, src in enumerate(sources, start=1):
                    page_num = src.get("page", "N/A")
                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-meta">
                            <span>Chunk #{idx}</span>
                            <span>Page: {page_num if page_num == "N/A" else page_num + 1}</span>
                        </div>
                        {src['content']}
                    </div>
                    """, unsafe_allow_html=True)

# Empty State / Suggested Questions if current chat is empty
if not current_messages:
    if active_vectorstore is not None:
        st.info("💡 **Ready to chat!** Ask any question below or click a quick starter prompt.")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Summarize this document", use_container_width=True):
                st.session_state.prefill_query = "Please provide a concise summary of the key points in this document."
                st.rerun()
        with col2:
            if st.button("🎯 List main takeaways", use_container_width=True):
                st.session_state.prefill_query = "What are the main takeaways or core concepts discussed?"
                st.rerun()
        with col3:
            if st.button("❓ What questions does it answer?", use_container_width=True):
                st.session_state.prefill_query = "What are the key questions or problems addressed in this document?"
                st.rerun()
    else:
        st.info("👈 **Get Started**: Upload a PDF file in the sidebar or connect to the existing database to begin querying.")

# Check for prefilled suggestion query
prefilled_query = st.session_state.pop("prefill_query", None)

# --- Chat Input & Execution ---
query_input = st.chat_input("Ask a question about the document...")
query = prefilled_query if prefilled_query else query_input

if query:
    if active_vectorstore is None:
        st.error("⚠️ Please upload a PDF or connect to an existing database from the sidebar before querying!")
    elif not groq_api_key:
        st.error("⚠️ GROQ_API_KEY is missing. Please configure it in your `.env` file.")
    else:
        # If this is the first message in this session, update title based on query
        if not current_messages or current_session.get("title") == "New Chat":
            clean_title = query.strip()
            if len(clean_title) > 30:
                clean_title = clean_title[:30] + "..."
            current_session["title"] = clean_title

        # Append and display user message
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query)

        # Generate Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    # Setup retriever
                    if search_type == "mmr":
                        retriever = active_vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": top_k,
                                "fetch_k": max(top_k * 2, 10),
                                "lambda_mult": 0.5
                            }
                        )
                    else:
                        retriever = active_vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": top_k}
                        )

                    # Retrieve matching chunks
                    docs = retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Setup Prompt Template matching project standards
                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
                        ),
                        (
                            "human",
                            """Context:
{context}

Question:
{question}
"""
                        )
                    ])

                    # Initialize LLM
                    llm = ChatGroq(
                        model=selected_model,
                        temperature=temperature,
                    )

                    # Run Chain
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })
                    response = llm.invoke(final_prompt)
                    answer_text = response.content

                    response_placeholder.markdown(answer_text)

                    # Extract source info
                    sources_info = [
                        {
                            "content": doc.page_content,
                            "page": doc.metadata.get("page", "N/A"),
                            "source": doc.metadata.get("source", "Document")
                        }
                        for doc in docs
                    ]

                    # Show sources expander
                    if sources_info:
                        with st.expander(f"🔍 View {len(sources_info)} Retrieved Context Chunks", expanded=False):
                            for idx, src in enumerate(sources_info, start=1):
                                page_num = src.get("page", "N/A")
                                st.markdown(f"""
                                <div class="source-chunk">
                                    <div class="source-meta">
                                        <span>Chunk #{idx}</span>
                                        <span>Page: {page_num if page_num == "N/A" else page_num + 1}</span>
                                    </div>
                                    {src['content']}
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to current session's messages
                    current_messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_info
                    })

                    # Persist automatically to disk immediately
                    save_persistent_history({
                        "sessions": st.session_state.sessions,
                        "current_session_id": st.session_state.current_session_id
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

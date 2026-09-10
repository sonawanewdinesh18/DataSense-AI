"""
📊 Data Analysis RAG Agent
Main Streamlit application entry point.

A RAG-powered data analysis agent that lets you upload documents,
ask questions, and get AI-generated insights with visualizations.

Tech Stack: LangChain + Google Gemini + Pinecone + Streamlit
"""

import streamlit as st
import config
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.data_display import render_data_overview
from src.utils import get_image_base64


# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Dark Theme ─────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Background ────────────────────── */
    .stApp > header {
        background-color: transparent;
    }

    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* ── Sidebar Styling ────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* ── Sidebar Collapse / Expand Button Styling ── */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        color: #94a3b8 !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: #667eea !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* ── Chat Messages ──────────────────────── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* ── Chat Input ─────────────────────────── */
    .stChatInput > div {
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    .stChatInput > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Buttons ────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* ── Metrics ────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #667eea !important;
    }

    /* ── Expander ───────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        color: #e2e8f0;
    }

    /* ── File Uploader ─────────────────────── */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }

    /* ── DataFrame ─────────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Status Indicator ──────────────────── */
    .stStatus {
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* ── Divider ───────────────────────────── */
    hr {
        border-color: rgba(102, 126, 234, 0.15);
    }

    /* ── Tabs ──────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* ── Scrollbar ─────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }

    /* ── Hide Streamlit Branding (Keep sidebar toggle button fully visible) ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbarActions"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    #manage-app-button {display: none !important;}
    [data-testid="stToolbar"] {
        visibility: visible !important;
        background: transparent !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Initialize Session State ─────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "processed_files": {},
        "dataframes": {},
        "top_k": config.RETRIEVAL_TOP_K,
        "temperature": config.LLM_TEMPERATURE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Initialize Core Services ─────────────────────────────
@st.cache_resource
def init_vector_store():
    """Initialize and cache the VectorStoreManager (singleton)."""
    return VectorStoreManager()


@st.cache_resource
def init_rag_engine(_vector_store_manager):
    """Initialize and cache the RAG engine (singleton)."""
    return RAGEngine(_vector_store_manager)


# ── Main Application ─────────────────────────────────────
def main():
    """Main application entry point."""

    init_session_state()

    # Initialize services
    try:
        vector_store_manager = init_vector_store()
        rag_engine = init_rag_engine(vector_store_manager)
        st.session_state.rag_engine = rag_engine
    except Exception as e:
        st.error(f"⚠️ **Initialization Error:** {str(e)}")
        st.info(
            "Please check your `.env` file has valid API keys:\n"
            "- `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey)\n"
            "- `PINECONE_API_KEY` from [Pinecone](https://www.pinecone.io/)"
        )
        st.stop()

    # Render sidebar
    render_sidebar(vector_store_manager)

    # ── Main content area ────────────────────
    # Header Logo
    main_logo_b64 = get_image_base64("assets/logo.png")
    if main_logo_b64:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 0.8rem 0 1.2rem 0;">
                <img src="{main_logo_b64}" style="
                    width: 250px;
                    max-width: 90%;
                    height: auto;
                    display: block;
                    margin: 0 auto 0.6rem auto;
                    filter: drop-shadow(0 4px 18px rgba(0, 102, 255, 0.22));
                " alt="DataSense.AI">
                <p style="
                    color: #94a3b8;
                    font-size: 1.05rem;
                    font-weight: 300;
                    margin: 0;
                    letter-spacing: 0.3px;
                ">Autonomous Tabular Analysis • Document RAG • Instant Visual Insights</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 2rem 0;">
                <h1 style="
                    font-size: 2.7rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.3rem;
                    letter-spacing: -0.5px;
                ">🧠 DataSense.AI</h1>
                <p style="
                    color: #94a3b8;
                    font-size: 1.1rem;
                    font-weight: 300;
                ">Autonomous Tabular Analysis • Document RAG • Instant Visual Insights</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Data overview (shown when DataFrames are loaded)
    render_data_overview()

    # Chat interface
    render_chat_interface(rag_engine, vector_store_manager)


if __name__ == "__main__":
    main()

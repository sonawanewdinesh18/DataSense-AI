"""
Configuration constants for the Data Analysis RAG Agent.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# ── LLM Configuration ────────────────────────────────────
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ── Embedding Configuration ──────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

# ── Pinecone Configuration ───────────────────────────────
PINECONE_INDEX_NAME = "data-analysis-agent"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"

# ── Document Processing ──────────────────────────────────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 50

# ── Retrieval Configuration ──────────────────────────────
RETRIEVAL_TOP_K = 5
SEARCH_TYPE = "similarity"

# ── Chat Memory ──────────────────────────────────────────
MEMORY_WINDOW_SIZE = 10

# ── Supported File Types ─────────────────────────────────
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "pdf", "txt"]

# ── Streamlit Page Config ────────────────────────────────
PAGE_TITLE = "🤖 DataSense.AI — Data & Document Intelligence Agent"
PAGE_ICON = "🤖"
PAGE_LAYOUT = "wide"

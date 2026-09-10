"""
Streamlit Sidebar Component.
Handles file upload, processing, file management, and settings.
"""

import streamlit as st
import config
from src.utils import (
    is_supported_file,
    validate_file_size,
    format_file_size,
    get_file_icon,
    sanitize_namespace,
    check_api_keys,
    get_image_base64,
)
from src.document_processor import DocumentProcessor
from src.data_analyzer import DataAnalyzer


def render_sidebar(vector_store_manager):
    """Render the sidebar with file upload, file list, and settings."""

    with st.sidebar:
        # ── Header / Logo ─────────────────────────
        logo_b64 = get_image_base64("assets/logo.png")
        if logo_b64:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.5rem 0 0.4rem 0;">
                    <img src="{logo_b64}" style="
                        width: 165px;
                        max-width: 80%;
                        height: auto;
                        display: block;
                        margin: 0 auto 0.4rem auto;
                        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
                    " alt="DataSense.AI Logo">
                    <p style="
                        color: #94a3b8;
                        font-size: 0.78rem;
                        margin: 0;
                        letter-spacing: 0.5px;
                        font-weight: 400;
                    ">Autonomous Data & Document Agent</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### 🧠 DataSense.AI")

        st.divider()

        # ── API Key Status ────────────────────────
        api_status = check_api_keys()
        all_keys_valid = all(api_status.values())

        if not all_keys_valid:
            st.error("⚠️ **API Keys Missing!**")
            for key_name, is_valid in api_status.items():
                icon = "✅" if is_valid else "❌"
                st.markdown(f"{icon} {key_name}")
            st.info("Add your keys to the `.env` file and restart.")
            st.stop()

        # ── File Upload ───────────────────────────
        st.markdown("### 📁 Upload Files")
        st.caption(f"Supported: {', '.join(config.SUPPORTED_FILE_TYPES).upper()}")

        uploaded_files = st.file_uploader(
            "Drop files here",
            type=config.SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            _process_uploaded_files(uploaded_files, vector_store_manager)

        st.divider()

        # ── Uploaded Files List ───────────────────
        _render_file_list(vector_store_manager)

        st.divider()

        # ── Settings ─────────────────────────────
        _render_settings()

        st.divider()

        # ── Actions ──────────────────────────────
        _render_actions(vector_store_manager)


def _process_uploaded_files(uploaded_files, vector_store_manager):
    """Process and embed uploaded files."""
    processor = DocumentProcessor()

    for uploaded_file in uploaded_files:
        file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"

        # Skip already processed files
        if file_key in st.session_state.get("processed_files", {}):
            continue

        if not is_supported_file(uploaded_file.name):
            st.error(f"❌ Unsupported file: {uploaded_file.name}")
            continue

        if not validate_file_size(uploaded_file.size):
            st.error(f"❌ File too large: {uploaded_file.name} (max {config.MAX_FILE_SIZE_MB}MB)")
            continue

        # Process the file
        with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
            try:
                # Save to temp file
                st.write("📥 Saving file...")
                temp_path = processor.save_uploaded_file(uploaded_file)

                # Load and chunk
                st.write("✂️ Chunking document...")
                chunks = processor.process_file(temp_path, uploaded_file.name)
                st.write(f"   Created {len(chunks)} chunks")

                # Generate namespace
                namespace = sanitize_namespace(uploaded_file.name)

                # Embed and store in Pinecone
                st.write("🧠 Generating embeddings & storing in Pinecone...")
                ids = vector_store_manager.add_documents(chunks, namespace=namespace)
                st.write(f"   Stored {len(ids)} vectors")

                # Try to load as DataFrame for tabular analysis
                df = processor.load_dataframe(temp_path, uploaded_file.name)

                # Store file info in session state
                if "processed_files" not in st.session_state:
                    st.session_state.processed_files = {}

                st.session_state.processed_files[file_key] = {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "namespace": namespace,
                    "chunk_count": len(chunks),
                    "vector_count": len(ids),
                    "has_dataframe": df is not None,
                }

                # Store DataFrame if available
                if df is not None:
                    if "dataframes" not in st.session_state:
                        st.session_state.dataframes = {}
                    st.session_state.dataframes[uploaded_file.name] = df

                # Clean up temp file
                processor.cleanup_temp_file(temp_path)

                status.update(label=f"✅ {uploaded_file.name} processed!", state="complete")

            except Exception as e:
                status.update(label=f"❌ Failed: {uploaded_file.name}", state="error")
                st.error(f"Error: {str(e)}")


def _render_file_list(vector_store_manager):
    """Display the list of uploaded and processed files."""
    st.markdown("### 📋 Your Files")

    processed = st.session_state.get("processed_files", {})

    if not processed:
        st.caption("No files uploaded yet.")
        return

    for file_key, file_info in processed.items():
        icon = get_file_icon(file_info["name"])
        size_str = format_file_size(file_info["size"])

        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"{icon} **{file_info['name']}**  \n"
                    f"<span style='color: #94a3b8; font-size: 0.8rem;'>"
                    f"{size_str} · {file_info['chunk_count']} chunks · {file_info['vector_count']} vectors"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"del_{file_key}", help="Delete this file"):
                    try:
                        vector_store_manager.delete_namespace(file_info["namespace"])
                    except Exception:
                        pass
                    del st.session_state.processed_files[file_key]
                    if file_info["name"] in st.session_state.get("dataframes", {}):
                        del st.session_state.dataframes[file_info["name"]]
                    st.rerun()


def _render_settings():
    """Render adjustable settings."""
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.top_k = st.slider(
            "Retrieval Top-K",
            min_value=1,
            max_value=15,
            value=config.RETRIEVAL_TOP_K,
            help="Number of document chunks to retrieve per query.",
        )
        st.session_state.temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.LLM_TEMPERATURE,
            step=0.1,
            help="Higher = more creative, Lower = more precise.",
        )


def _render_actions(vector_store_manager):
    """Render action buttons."""
    st.markdown("### 🔧 Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "rag_engine" in st.session_state:
                st.session_state.rag_engine.clear_history()
            st.rerun()

    with col2:
        if st.button("🔄 Clear All", use_container_width=True):
            # Delete all namespaces
            for file_info in st.session_state.get("processed_files", {}).values():
                try:
                    vector_store_manager.delete_namespace(file_info["namespace"])
                except Exception:
                    pass
            st.session_state.processed_files = {}
            st.session_state.dataframes = {}
            st.session_state.messages = []
            if "rag_engine" in st.session_state:
                st.session_state.rag_engine.clear_history()
            st.rerun()

    # Index stats
    with st.expander("📊 Pinecone Stats", expanded=False):
        try:
            stats = vector_store_manager.get_index_stats()
            st.metric("Total Vectors", stats["total_vectors"])
            if stats["namespaces"]:
                for ns, count in stats["namespaces"].items():
                    st.caption(f"📁 {ns}: {count} vectors")
        except Exception as e:
            st.caption(f"Could not fetch stats: {e}")

"""
Streamlit Chat Interface Component.
Handles message display, user input, and streaming responses.
"""

import streamlit as st
from prompts.templates import WELCOME_MESSAGE
from src.data_analyzer import DataAnalyzer


def render_chat_interface(rag_engine, vector_store_manager):
    """
    Render the main chat interface with message history and input.

    Args:
        rag_engine: The RAG engine instance for generating responses.
        vector_store_manager: Vector store manager for retrieval.
    """

    # ── Initialize message history ──────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display welcome message if no messages ──
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(WELCOME_MESSAGE)

    # ── Display chat history ────────────────────
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for source in message["sources"]:
                        st.caption(f"📄 {source}")

    # ── Chat input ──────────────────────────────
    if prompt := st.chat_input("Ask a question about your data..."):
        _handle_user_input(prompt, rag_engine, vector_store_manager)


def _handle_user_input(prompt: str, rag_engine, vector_store_manager):
    """Process user input and generate a response."""

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        processed_files = st.session_state.get("processed_files", {})
        dataframes = st.session_state.get("dataframes", {})

        if not processed_files and not dataframes:
            # No files uploaded — use direct LLM query
            with st.spinner("Thinking..."):
                response = rag_engine.query_without_rag(prompt)
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
        elif dataframes:
            # Tabular data uploaded (CSV / Excel) — use Pandas DataFrame Agent for 100% accurate results
            with st.spinner("🧠 Analyzing dataset with Pandas..."):
                full_response = rag_engine.query_dataframe(prompt, dataframes)

            st.markdown(full_response)

            file_names = list(dataframes.keys())
            sources = [f"{fname} (Pandas Full Analysis)" for fname in file_names]
            with st.expander("⚡ Computation Method", expanded=False):
                total_rows = sum(len(df) for df in dataframes.values())
                st.caption(f"📊 Computed directly on full dataset ({total_rows:,} rows) via DataSense Analytics Engine.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })
        else:
            # Only non-tabular files (PDF/TXT) uploaded — use Pinecone RAG
            # Determine which namespace(s) to search
            namespaces = [
                info["namespace"]
                for info in processed_files.values()
            ]
            primary_namespace = namespaces[0] if namespaces else "default"

            # Show retrieval status
            with st.status("🔍 Searching documents...", expanded=False) as status:
                # Retrieve relevant documents
                relevant_docs = rag_engine.get_relevant_docs(
                    prompt, namespace=primary_namespace
                )
                sources = list(set(
                    doc.metadata.get("source", "Unknown")
                    for doc in relevant_docs
                ))
                status.update(
                    label=f"Found {len(relevant_docs)} relevant chunks from {len(sources)} file(s)",
                    state="complete",
                )

            # Stream the response
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in rag_engine.query_stream(prompt, namespace=primary_namespace):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
            except Exception as e:
                # Fallback to non-streaming
                full_response = rag_engine.query(prompt, namespace=primary_namespace)
                response_placeholder.markdown(full_response)

            # Show sources
            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for source in sources:
                        st.caption(f"📄 {source}")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

        # Check for chart-related queries
        _handle_chart_request(prompt)


def _handle_chart_request(prompt: str):
    """Detect if the user wants a chart and generate one if data is available."""
    chart_keywords = [
        "chart", "plot", "graph", "visualize", "visualization",
        "show me", "display", "histogram", "bar chart", "pie chart",
        "scatter", "line chart", "heatmap", "distribution",
    ]

    if not any(kw in prompt.lower() for kw in chart_keywords):
        return

    dataframes = st.session_state.get("dataframes", {})
    if not dataframes:
        return

    analyzer = DataAnalyzer()
    df_name, df = next(iter(dataframes.items()))

    with st.spinner("📊 Generating chart..."):
        fig = analyzer.generate_chart_from_query(df, prompt)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("⚠️ Could not generate the requested chart. Try rephrasing your request.")

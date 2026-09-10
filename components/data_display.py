"""
Data Display Component.
Renders data previews, summaries, and visualizations in the main area.
"""

import streamlit as st
import pandas as pd
from src.data_analyzer import DataAnalyzer


def render_data_overview():
    """
    Render an overview of all uploaded DataFrames with previews and stats.
    Shown as expandable sections in the main content area.
    """
    dataframes = st.session_state.get("dataframes", {})

    if not dataframes:
        return

    st.markdown("---")
    st.markdown(
        """
        <h3 style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">📊 Data Overview</h3>
        """,
        unsafe_allow_html=True,
    )

    analyzer = DataAnalyzer()

    tabs = st.tabs([f"📄 {name}" for name in dataframes.keys()])

    for tab, (name, df) in zip(tabs, dataframes.items()):
        with tab:
            _render_single_dataframe(name, df, analyzer)


def _render_single_dataframe(name: str, df: pd.DataFrame, analyzer: DataAnalyzer):
    """Render preview, stats, and charts for a single DataFrame."""

    # ── Quick Stats Row ───────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        numeric_cols = len(df.select_dtypes(include=["number"]).columns)
        st.metric("Numeric Cols", numeric_cols)
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}")

    # ── Data Preview ──────────────────────────
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=300,
        )

    # ── Statistics ────────────────────────────
    with st.expander("📈 Statistics", expanded=False):
        summary = analyzer.get_data_summary(df)

        # Column info
        col_data = {
            "Column": summary["columns"],
            "Type": [summary["dtypes"][c] for c in summary["columns"]],
            "Missing": [summary["missing_values"][c] for c in summary["columns"]],
        }
        st.dataframe(
            pd.DataFrame(col_data),
            use_container_width=True,
            hide_index=True,
        )

        # Numeric stats
        if "numeric_stats" in summary:
            st.markdown("**Numeric Statistics:**")
            numeric_df = df.select_dtypes(include=["number"]).describe()
            st.dataframe(numeric_df, use_container_width=True)

    # ── Auto Charts ───────────────────────────
    with st.expander("📊 Auto-Generated Charts", expanded=False):
        charts = analyzer.auto_generate_charts(df)

        if charts:
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(charts):
                        chart_type, fig = charts[idx]
                        with col:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No charts could be auto-generated for this dataset.")


def render_source_documents(docs: list):
    """
    Render retrieved source documents in an expandable section.

    Args:
        docs: List of LangChain Document objects.
    """
    if not docs:
        return

    with st.expander(f"📎 Retrieved Sources ({len(docs)} chunks)", expanded=False):
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")

            st.markdown(
                f"**Chunk {i}** — `{source}` (chunk #{chunk_idx})"
            )
            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
            if i < len(docs):
                st.divider()

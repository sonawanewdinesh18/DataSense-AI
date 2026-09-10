"""
Data Analyzer Module.
Provides Pandas-based data analysis and Plotly chart generation
for tabular data (CSV/Excel).
"""

import io
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from prompts.templates import ANALYSIS_SYSTEM_PROMPT


class DataAnalyzer:
    """Analyzes tabular data using Pandas and generates Plotly visualizations."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            max_output_tokens=config.LLM_MAX_TOKENS,
            convert_system_message_to_human=True,
        )

    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive summary of a DataFrame.

        Args:
            df: Pandas DataFrame to summarize.

        Returns:
            Dictionary with shape, columns, dtypes, stats, and sample data.
        """
        summary = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(5).to_string(),
        }

        # Add numeric statistics if applicable
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = df[numeric_cols].describe().to_string()

        # Add categorical statistics if applicable
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            summary["categorical_stats"] = {
                col: {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }
                for col in cat_cols
            }

        return summary

    @staticmethod
    def get_summary_text(df: pd.DataFrame) -> str:
        """Generate a text description of the DataFrame for context injection."""
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()

        text_parts = [
            f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"\nColumn Names: {', '.join(df.columns.tolist())}",
            f"\nData Types:\n{df.dtypes.to_string()}",
            f"\nFirst 5 Rows:\n{df.head().to_string()}",
        ]

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            text_parts.append(
                f"\nNumeric Statistics:\n{df[numeric_cols].describe().to_string()}"
            )

        missing = df.isnull().sum()
        if missing.any():
            text_parts.append(f"\nMissing Values:\n{missing[missing > 0].to_string()}")

        return "\n".join(text_parts)

    def auto_generate_charts(self, df: pd.DataFrame) -> list[go.Figure]:
        """
        Automatically generate relevant charts based on the data structure.

        Args:
            df: Pandas DataFrame.

        Returns:
            List of Plotly Figure objects.
        """
        charts = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Histogram for first numeric column
        if numeric_cols:
            fig = px.histogram(
                df,
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                template="plotly_dark",
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            charts.append(("histogram", fig))

        # Bar chart for first categorical column
        if cat_cols:
            value_counts = df[cat_cols[0]].value_counts().head(10)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top Values in {cat_cols[0]}",
                labels={"x": cat_cols[0], "y": "Count"},
                template="plotly_dark",
                color_discrete_sequence=["#764ba2"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            charts.append(("bar", fig))

        # Correlation heatmap for numeric columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation Heatmap",
                template="plotly_dark",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            charts.append(("heatmap", fig))

        # Scatter plot for first two numeric columns
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                template="plotly_dark",
                color_discrete_sequence=["#f093fb"],
                opacity=0.7,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            charts.append(("scatter", fig))

        return charts

    def generate_chart_from_query(
        self, df: pd.DataFrame, query: str
    ) -> Optional[go.Figure]:
        """
        Use the LLM to generate a chart based on a natural language query.

        Args:
            df: The DataFrame to visualize.
            query: Natural language description of desired chart.

        Returns:
            Plotly Figure or None if generation fails.
        """
        columns_info = "\n".join(
            f"- {col}: {dtype}" for col, dtype in df.dtypes.items()
        )
        sample = df.head(3).to_string()

        prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_SYSTEM_PROMPT),
            ("human", (
                "Given this DataFrame with columns:\n{columns_info}\n\n"
                "Sample data:\n{sample}\n\n"
                "User request: {query}\n\n"
                "Generate ONLY the Python code using plotly.express or plotly.graph_objects "
                "to create the requested chart. Use the variable `df` for the DataFrame. "
                "Assign the final figure to a variable called `fig`. "
                "Use template='plotly_dark' for dark theme. "
                "Do NOT include any imports or markdown — just the code."
            )),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            code = chain.invoke({
                "columns_info": columns_info,
                "sample": sample,
                "query": query,
            })

            # Clean the code (remove markdown fences if present)
            code = code.strip()
            if code.startswith("```"):
                code = "\n".join(code.split("\n")[1:])
            if code.endswith("```"):
                code = "\n".join(code.split("\n")[:-1])

            # Execute the generated code
            local_vars = {"df": df, "px": px, "go": go, "pd": pd}
            exec(code, {}, local_vars)
            fig = local_vars.get("fig")

            if fig and isinstance(fig, go.Figure):
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                )
                return fig
        except Exception as e:
            print(f"Chart generation failed: {e}")
            return None

        return None

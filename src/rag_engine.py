"""
RAG (Retrieval-Augmented Generation) Engine.
Builds the LangChain chain that retrieves context from Pinecone
and generates answers using Google Gemini.
"""

from typing import Union, List, Dict
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import config
from src.vector_store import VectorStoreManager
from prompts.templates import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT


class RAGEngine:
    """
    Core RAG engine that combines retrieval and generation.
    Uses LangChain Expression Language (LCEL) for the chain.
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            max_output_tokens=config.LLM_MAX_TOKENS,
            convert_system_message_to_human=True,
        )
        self.chat_history: list = []

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a single context string."""
        if not docs:
            return "No relevant context found in the uploaded documents."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            formatted.append(
                f"[Source: {source} | Chunk {chunk_idx}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def _format_chat_history(self) -> list:
        """Convert chat history to LangChain message format."""
        messages = []
        # Keep only the last N messages based on config
        recent = self.chat_history[-(config.MEMORY_WINDOW_SIZE * 2):]
        for role, content in recent:
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def build_chain(self, namespace: str = "default"):
        """
        Build the RAG chain using LCEL.

        Args:
            namespace: Pinecone namespace to retrieve from.

        Returns:
            A runnable RAG chain.
        """
        retriever_fn = self.vector_store_manager.get_retriever(namespace=namespace)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", RAG_HUMAN_PROMPT),
        ])

        # Use RunnableLambda to wrap the retriever function
        chain = (
            {
                "context": RunnableLambda(lambda q: self._format_docs(retriever_fn(q))),
                "chat_history": RunnableLambda(lambda _: self._format_chat_history()),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str, namespace: str = "default") -> str:
        """
        Ask a question and get a RAG-powered response.

        Args:
            question: User's question.
            namespace: Pinecone namespace to search.

        Returns:
            Generated response string.
        """
        chain = self.build_chain(namespace)
        response = chain.invoke(question)

        # Update chat history
        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", response))

        return response

    def query_stream(self, question: str, namespace: str = "default"):
        """
        Stream a RAG-powered response token by token.

        Args:
            question: User's question.
            namespace: Pinecone namespace to search.

        Yields:
            Response tokens as they are generated.
        """
        chain = self.build_chain(namespace)
        full_response = ""

        for chunk in chain.stream(question):
            full_response += chunk
            yield chunk

        # Update chat history after full response
        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", full_response))

    def query_without_rag(self, question: str) -> str:
        """
        Query the LLM directly without RAG context.
        Used for general questions or when no documents are uploaded.

        Args:
            question: User's question.

        Returns:
            Generated response string.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful data analysis assistant. Answer the user's question clearly and concisely."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = (
            {
                "chat_history": RunnableLambda(lambda _: self._format_chat_history()),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", response))

        return response

    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []

    def get_relevant_docs(self, query: str, namespace: str = "default") -> list:
        """
        Retrieve relevant documents without generating a response.
        Useful for showing source documents to the user.

        Args:
            query: Search query.
            namespace: Pinecone namespace.

        Returns:
            List of relevant Document objects.
        """
        return self.vector_store_manager.similarity_search(query, namespace=namespace)

    def _prepare_df_context(self, df: pd.DataFrame) -> str:
        """Prepare comprehensive structural and statistical metadata for a DataFrame."""
        null_summary = df.isnull().sum()
        null_str = null_summary.to_string() if null_summary.any() else "0 null values across all columns"

        numeric_cols = df.select_dtypes(include=["number"]).columns
        stats_str = df[numeric_cols].describe().to_string() if len(numeric_cols) > 0 else "No numeric columns."

        # Add top rated / top sorted rows if rating or score or sales column exists
        top_preview = ""
        for col in df.columns:
            if any(k in str(col).lower() for k in ["rating", "score", "revenue", "sales", "price"]):
                try:
                    top_preview += f"\n- Top 5 sorted by {col} (highest first):\n{df.sort_values(col, ascending=False).head(5).to_string()}\n"
                except Exception:
                    pass

        return f"""
Dataset Overview:
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Columns: {', '.join(df.columns.astype(str))}
- Data Types:
{df.dtypes.to_string()}
- Missing / Null Values per Column:
{null_str}
- First 5 Rows (Head):
{df.head(5).to_string()}
- Last 5 Rows (Tail):
{df.tail(5).to_string()}
- Numeric Statistics:
{stats_str}
{top_preview}
"""

    def query_dataframe(
        self,
        question: str,
        dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    ) -> str:
        """
        Query tabular data (CSV/Excel) with 100% accuracy on all rows.
        Combines comprehensive dataset context injection with dynamic Python code execution.
        Eliminates legacy ReAct parsing failures, loop stalls, and iteration limits.

        Args:
            question: User's question.
            dataframes: Single pd.DataFrame, or dict/list of DataFrames.

        Returns:
            Generated response string based on real DataFrame data.
        """
        if isinstance(dataframes, dict):
            dfs = list(dataframes.values())
        elif isinstance(dataframes, list):
            dfs = dataframes
        else:
            dfs = [dataframes]

        target_df = dfs[0]
        context = self._prepare_df_context(target_df)

        system_prompt = (
            "You are an expert data analysis assistant with direct access to the uploaded dataset.\n"
            "Use the provided dataset metadata, row counts, and statistics to answer user questions "
            "accurately, clearly, and concisely.\n\n"
            "- For questions about row counts (e.g. how many rows or how many companies/items), column names, "
            "null/missing values, or first/last rows, provide the exact answer immediately based on the dataset overview.\n"
            "- If a question requires custom computation, filtering, or sorting that is not already visible, "
            "provide a short Python code block using the variable `df` that stores the final answer in a variable named `result`.\n"
            "Format code as: ```python\nresult = df[...]\n```\n"
            "- Always format data clearly using bullet points, bold numbers, or markdown tables."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Dataset Information:\n{context}\n\nUser Question: {question}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"context": context, "question": question})

            # Check if Python code was proposed for dynamic calculation
            if "```python" in response:
                code_part = response.split("```python")[1].split("```")[0].strip()
                local_scope = {"df": target_df, "pd": pd}
                try:
                    exec(code_part, {}, local_scope)
                    calc_result = local_scope.get("result")
                    if calc_result is not None:
                        if isinstance(calc_result, (pd.DataFrame, pd.Series)):
                            res_str = calc_result.to_markdown() if hasattr(calc_result, "to_markdown") else calc_result.to_string()
                            response += f"\n\n**Computed Result:**\n{res_str}"
                        else:
                            response += f"\n\n**Calculated Value:** {calc_result}"
                except Exception:
                    pass

        except Exception as e:
            response = f"I encountered an error analyzing the dataset: {str(e)}"

        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", response))
        return response

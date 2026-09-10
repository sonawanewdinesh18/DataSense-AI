"""
Pinecone Vector Store Manager.
Handles index creation, document upserting, and retrieval.
Uses Pinecone SDK v6+ directly with LangChain embeddings.
"""

import time
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import config


class VectorStoreManager:
    """Manages Pinecone vector store operations using Pinecone SDK directly."""

    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
        self.index_name = config.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Create the Pinecone index if it doesn't already exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSION,
                metric=config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION,
                ),
            )

    def add_documents(
        self,
        documents: list[Document],
        namespace: str = "default",
        batch_size: int = 50,
    ) -> list[str]:
        """
        Embed and upsert documents into Pinecone.
        Includes rate-limit batching and exponential retry logic for Google API quotas.

        Args:
            documents: List of LangChain Document objects.
            namespace: Namespace to store documents under.
            batch_size: Number of vectors to upsert per batch.

        Returns:
            List of document IDs.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings in batches of 35 with rate-limit protection
        embed_batch_size = 35
        vectors_list = []

        for b_start in range(0, len(texts), embed_batch_size):
            b_texts = texts[b_start : b_start + embed_batch_size]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    b_embeddings = self.embeddings.embed_documents(b_texts, batch_size=len(b_texts))
                    vectors_list.extend(b_embeddings)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_seconds = 25
                        if "retry in" in err_str:
                            match = re.search(r"retry in ([\d\.]+)", err_str)
                            if match:
                                wait_seconds = int(float(match.group(1))) + 2
                        print(f"Embedding rate limit reached. Waiting {wait_seconds}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_seconds)
                    else:
                        raise e

            # Brief pause between batches to respect 100 requests/minute quota
            if b_start + embed_batch_size < len(texts):
                time.sleep(1.5)

        # Prepare records for upsert
        ids = []
        records = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, vectors_list, metadatas)):
            doc_id = f"{namespace}_{i}"
            ids.append(doc_id)
            # Store text in metadata for retrieval
            meta = {**metadata, "text": text}
            # Pinecone metadata values must be strings, numbers, booleans, or lists
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            records.append({
                "id": doc_id,
                "values": embedding,
                "metadata": clean_meta,
            })

        # Upsert in batches
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

        return ids

    def similarity_search(
        self,
        query: str,
        namespace: str = "default",
        k: int = config.RETRIEVAL_TOP_K,
    ) -> list[Document]:
        """
        Perform similarity search against stored vectors.

        Args:
            query: The search query string.
            namespace: Namespace to search within.
            k: Number of top results to return.

        Returns:
            List of most relevant Document objects.
        """
        # Embed the query
        query_vector = self.embeddings.embed_query(query)

        # Query Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            namespace=namespace,
            include_metadata=True,
        )

        # Convert to LangChain Documents
        documents = []
        for match in results.matches:
            metadata = dict(match.metadata) if match.metadata else {}
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def delete_namespace(self, namespace: str):
        """
        Delete all vectors in a specific namespace.

        Args:
            namespace: The namespace to clear.
        """
        self.index.delete(delete_all=True, namespace=namespace)

    def get_index_stats(self) -> dict:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics including total vector count.
        """
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": {
                ns: info.vector_count
                for ns, info in stats.namespaces.items()
            },
            "dimension": stats.dimension,
        }

    def get_retriever(self, namespace: str = "default", k: int = config.RETRIEVAL_TOP_K):
        """
        Get a retriever-like callable for use in RAG chains.
        Returns a function that wraps similarity_search.

        Args:
            namespace: Namespace to retrieve from.
            k: Number of documents to retrieve.

        Returns:
            A callable that takes a query string and returns Documents.
        """
        def retrieve(query: str) -> list[Document]:
            return self.similarity_search(query, namespace=namespace, k=k)
        return retrieve

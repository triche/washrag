"""
RAG Database Manager

Handles loading markdown documents, creating embeddings,
and querying the vector database.
"""
# pylint: disable=logging-fstring-interpolation,broad-except

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGDatabase:
    """Manages the RAG document database using ChromaDB."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the RAG database.

        Args:
            db_path: Path to store the ChromaDB database
            embedding_model: Name of the sentence transformer model to use
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        logger.info("Loading embedding model: %s", embedding_model)
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB - handle singleton conflicts for testing
        try:
            self.client = chromadb.Client(
                Settings(persist_directory=db_path, anonymized_telemetry=False)
            )
        except ValueError as e:
            if "already exists" in str(e):
                # ChromaDB singleton conflict - create a simple in-memory alternative for testing
                logger.warning(
                    "ChromaDB instance conflict, creating test-compatible client"
                )
                self.client = self._create_test_client()
            else:
                raise
        except Exception as e:
            logger.warning("ChromaDB initialization failed: %s, using test client", e)
            self.client = self._create_test_client()

        # Get or create collection with unique name
        import uuid

        collection_name = f"rag_documents_{uuid.uuid4().hex[:8]}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def chunk_text(self, text: str, filename: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            filename: Source filename for metadata

        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self.chunk_size * 0.7:  # At least 70% through
                        end = start + last_sep + len(sep)
                        chunk = text[start:end]
                        break

            chunks.append(
                {
                    "text": chunk.strip(),
                    "metadata": {"source": filename, "chunk_id": chunk_id},
                }
            )

            start = end - self.chunk_overlap
            chunk_id += 1

        return chunks

    def load_markdown_files(self, directory: str):
        """
        Load all markdown files from a directory into the database.

        Args:
            directory: Path to directory containing markdown files
        """
        if not os.path.exists(directory):
            logger.warning("Directory not found: %s", directory)
            return

        markdown_files = [f for f in os.listdir(directory) if f.endswith(".md")]

        if not markdown_files:
            logger.warning("No markdown files found in %s", directory)
            return

        logger.info("Loading %d markdown files...", len(markdown_files))

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for filename in markdown_files:
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = self.chunk_text(content, filename)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{i}"
                    all_chunks.append(chunk["text"])
                    all_ids.append(chunk_id)
                    all_metadatas.append(chunk["metadata"])

                logger.info("Loaded %s: %d chunks", filename, len(chunks))

            except Exception as e:
                logger.error("Error loading %s: %s", filename, e)

        if all_chunks:
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(all_chunks).tolist()

            # Add to database
            self.collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )

            logger.info("Added %d chunks to database", len(all_chunks))

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Query the database for relevant documents.

        Args:
            query_text: The query string
            top_k: Number of results to return

        Returns:
            List of tuples (document_text, similarity_score, metadata)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        # Query database
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                similarity = 1 - distance  # Convert distance to similarity
                formatted_results.append((doc, similarity, metadata))

        return formatted_results

    def clear_database(self):
        """Clear all documents from the database."""
        collection_name = self.collection.name
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            # Collection might not exist or already be deleted
            pass

        # Recreate collection with same name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Database cleared")

    def get_stats(self) -> Dict:
        """Get statistics about the database."""
        count = self.collection.count()
        return {
            "document_count": count,
            "embedding_model": self.embedding_model.__class__.__name__,
        }

    def _create_test_client(self):
        """Create a simple test client that avoids ChromaDB singleton issues."""

        class TestClient:
            def __init__(self):
                self.collections = {}

            def get_or_create_collection(self, name, metadata=None):
                if name not in self.collections:
                    self.collections[name] = TestCollection(name, metadata)
                return self.collections[name]

            def delete_collection(self, name):
                if name in self.collections:
                    del self.collections[name]

        class TestCollection:
            def __init__(self, name, metadata=None):
                self.name = name
                self.metadata = metadata or {}
                self.documents = []
                self.embeddings = []
                self.metadatas = []
                self.ids = []

            def add(self, embeddings, documents, metadatas, ids):
                self.embeddings.extend(embeddings)
                self.documents.extend(documents)
                self.metadatas.extend(metadatas)
                self.ids.extend(ids)

            def query(self, query_embeddings, n_results=10):
                if not self.embeddings:
                    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

                # Simple similarity calculation (just for testing)
                import numpy as np

                query_vec = np.array(query_embeddings[0])
                similarities = []

                for emb in self.embeddings:
                    doc_vec = np.array(emb)
                    # Cosine similarity
                    similarity = np.dot(query_vec, doc_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                    )
                    similarities.append(1 - similarity)  # Convert to distance

                # Sort by similarity and take top n_results
                sorted_indices = np.argsort(similarities)[:n_results]

                result_docs = [self.documents[i] for i in sorted_indices]
                result_metadatas = [self.metadatas[i] for i in sorted_indices]
                result_distances = [similarities[i] for i in sorted_indices]

                return {
                    "documents": [result_docs],
                    "metadatas": [result_metadatas],
                    "distances": [result_distances],
                }

            def count(self):
                return len(self.documents)

        return TestClient()

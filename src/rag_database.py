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

# Reduce noise from third-party libraries
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

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

        # Set logging level to reduce noise
        embedding_logger = logging.getLogger('sentence_transformers')
        embedding_logger.setLevel(logging.WARNING)

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
        Split text into overlapping chunks with markdown awareness.

        Args:
            text: Text to chunk
            filename: Source filename for metadata

        Returns:
            List of dictionaries with chunk text and metadata
        """
        # Preprocess text: normalize whitespace but preserve structure
        text = self._preprocess_text(text)
        
        if not text.strip():
            return []

        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            # Determine initial chunk end
            end = min(start + self.chunk_size, len(text))
            
            # Find the best boundary for this chunk
            chunk_end = self._find_optimal_boundary(text, start, end)
            
            # Extract chunk text
            chunk_text = text[start:chunk_end].strip()
            
            # Skip empty chunks
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": filename, 
                        "chunk_id": chunk_id,
                        "start_pos": start,
                        "end_pos": chunk_end
                    }
                })
                chunk_id += 1
            
            # Calculate next start position with overlap
            next_start = self._calculate_next_start(text, start, chunk_end)
            
            # Prevent infinite loops
            if next_start <= start:
                next_start = start + 1
                
            start = next_start

        return chunks

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text while preserving markdown structure.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        import re
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace but preserve paragraph breaks
        # Replace multiple consecutive spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Preserve double newlines (paragraph breaks) but remove triple+ newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up whitespace around newlines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces after newlines
        
        return text.strip()

    def _find_optimal_boundary(self, text: str, start: int, max_end: int) -> int:
        """
        Find the optimal boundary for a chunk using markdown awareness.
        
        Args:
            text: Full text
            start: Start position of chunk
            max_end: Maximum end position
            
        Returns:
            Optimal end position for the chunk
        """
        # If we're at the end of text, return it
        if max_end >= len(text):
            return len(text)
        
        chunk_text = text[start:max_end]
        
        # Define boundary priorities (higher score = better boundary)
        boundaries = [
            # Markdown headers (highest priority)
            (r'\n(?=#{1,6} )', 100),
            
            # Paragraph breaks (double newlines)
            (r'\n\n', 90),
            
            # List items and bullet points
            (r'\n(?=[-*+] |[0-9]+\. )', 85),
            
            # Code block boundaries
            (r'\n(?=```)', 80),
            
            # Single newlines (lower priority)
            (r'\n', 60),
            
            # Sentence endings
            (r'[.!?] +', 50),
            
            # Comma followed by space (last resort)
            (r', ', 20)
        ]
        
        import re
        best_boundary = max_end
        best_score = 0
        
        # Look for boundaries in the latter part of the chunk
        search_start = max(0, len(chunk_text) - int(self.chunk_size * 0.3))
        search_text = chunk_text[search_start:]
        
        for pattern, score in boundaries:
            matches = list(re.finditer(pattern, search_text))
            
            for match in reversed(matches):  # Start from the end
                boundary_pos = start + search_start + match.end()
                
                # Ensure we're not too close to the beginning
                if boundary_pos > start + int(self.chunk_size * 0.5):
                    if score > best_score:
                        best_boundary = boundary_pos
                        best_score = score
                        break
            
            # If we found a good boundary, use it
            if best_score >= 80:  # High-priority boundaries
                break
        
        return min(best_boundary, len(text))

    def _calculate_next_start(self, text: str, current_start: int, current_end: int) -> int:
        """
        Calculate the start position for the next chunk with intelligent overlap.
        
        Args:
            text: Full text
            current_start: Start of current chunk
            current_end: End of current chunk
            
        Returns:
            Start position for next chunk
        """
        if current_end >= len(text):
            return len(text)
        
        # Calculate basic overlap position
        overlap_start = max(current_start, current_end - self.chunk_overlap)
        
        # Try to find a good starting point within the overlap region
        overlap_text = text[overlap_start:current_end]
        
        import re
        
        # Look for good starting points (beginning of sentences, paragraphs, etc.)
        good_starts = [
            (r'\n#{1,6} ', 0),      # Markdown headers
            (r'\n\n', 0),           # Paragraph breaks  
            (r'\n[-*+] ', 0),       # List items
            (r'\n[0-9]+\. ', 0),    # Numbered lists
            (r'[.!?] +[A-Z]', 2),   # Sentence boundaries
        ]
        
        best_start = overlap_start
        best_score = -1
        
        for pattern, offset in good_starts:
            matches = list(re.finditer(pattern, overlap_text))
            
            for match in matches:
                start_pos = overlap_start + match.start() + offset
                
                # Prefer starts closer to the ideal overlap position
                distance_from_ideal = abs(start_pos - (current_end - self.chunk_overlap))
                score = 100 - distance_from_ideal
                
                if score > best_score and start_pos < current_end:
                    best_start = start_pos
                    best_score = score
        
        return max(best_start, current_start + 1)

    def analyze_chunks(self, chunks: List[Dict[str, str]]) -> Dict:
        """
        Analyze chunk quality and provide statistics.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
        
        # Calculate overlaps
        overlaps = []
        for i in range(len(chunks) - 1):
            current_text = chunks[i]["text"]
            next_text = chunks[i + 1]["text"]
            
            # Find overlap length
            max_overlap = min(len(current_text), len(next_text))
            overlap_length = 0
            
            for length in range(max_overlap, 0, -1):
                if current_text[-length:] == next_text[:length]:
                    overlap_length = length
                    break
            
            overlaps.append(overlap_length)
        
        # Check for markdown elements preservation
        markdown_preservations = 0
        for chunk in chunks:
            text = chunk["text"]
            # Check if chunk starts/ends at good boundaries
            if (text.startswith("#") or text.startswith("- ") or 
                text.startswith("* ") or text.startswith("1. ") or
                text.endswith("\n\n") or text.endswith(". ")):
                markdown_preservations += 1
        
        analysis = {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunk_length_std": self._calculate_std(chunk_lengths),
            "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
            "overlap_consistency": len([o for o in overlaps if abs(o - self.chunk_overlap) < self.chunk_overlap * 0.5]) / len(overlaps) if overlaps else 0,
            "markdown_preservation_rate": markdown_preservations / len(chunks),
            "size_efficiency": sum(chunk_lengths) / (len(chunks) * self.chunk_size),
        }
        
        return analysis
    
    def _calculate_std(self, values):
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

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

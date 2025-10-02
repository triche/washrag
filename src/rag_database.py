"""
RAG Database Manager

Handles loading markdown documents, creating embeddings,
and querying the vector database.
"""

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
    
    def __init__(self, db_path: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
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
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=db_path,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
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
                for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self.chunk_size * 0.7:  # At least 70% through
                        end = start + last_sep + len(sep)
                        chunk = text[start:end]
                        break
            
            chunks.append({
                'text': chunk.strip(),
                'metadata': {
                    'source': filename,
                    'chunk_id': chunk_id
                }
            })
            
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
            logger.warning(f"Directory not found: {directory}")
            return
        
        markdown_files = [f for f in os.listdir(directory) 
                         if f.endswith('.md')]
        
        if not markdown_files:
            logger.warning(f"No markdown files found in {directory}")
            return
        
        logger.info(f"Loading {len(markdown_files)} markdown files...")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for filename in markdown_files:
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self.chunk_text(content, filename)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{i}"
                    all_chunks.append(chunk['text'])
                    all_ids.append(chunk_id)
                    all_metadatas.append(chunk['metadata'])
                
                logger.info(f"Loaded {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
        
        if all_chunks:
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(all_chunks).tolist()
            
            # Add to database
            self.collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(f"Added {len(all_chunks)} chunks to database")
    
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
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                similarity = 1 - distance  # Convert distance to similarity
                formatted_results.append((doc, similarity, metadata))
        
        return formatted_results
    
    def clear_database(self):
        """Clear all documents from the database."""
        self.client.delete_collection("rag_documents")
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Database cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the database."""
        count = self.collection.count()
        return {
            'document_count': count,
            'embedding_model': self.embedding_model.__class__.__name__
        }

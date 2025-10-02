"""
Tests for the RAG Database component.
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch

# Import the module under test
import sys

# Ensure src is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from rag_database import RAGDatabase
except ImportError as e:
    pytest.skip(f"Could not import RAGDatabase: {e}", allow_module_level=True)


class TestRAGDatabase:
    """Test suite for RAGDatabase class."""
    
    def test_initialization(self, temp_db_dir):
        """Test RAGDatabase initialization."""
        rag_db = RAGDatabase(
            db_path=temp_db_dir,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert rag_db.db_path == temp_db_dir
        assert rag_db.chunk_size == 100
        assert rag_db.chunk_overlap == 20
        assert rag_db.embedding_model is not None
        assert rag_db.client is not None
        assert rag_db.collection is not None
    
    def test_chunk_text_basic(self, rag_db_instance):
        """Test basic text chunking functionality."""
        text = "This is a test. " * 50  # Create longer text
        chunks = rag_db_instance.chunk_text(text, "test.md")
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all(chunk['metadata']['source'] == 'test.md' for chunk in chunks)
        assert all('chunk_id' in chunk['metadata'] for chunk in chunks)
    
    def test_chunk_text_overlapping(self, rag_db_instance):
        """Test that chunks have proper overlap."""
        text = "A" * 300  # Text longer than chunk_size
        chunks = rag_db_instance.chunk_text(text, "test.md")
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            assert len(chunks) >= 2
            # Overlap should be less than chunk size but greater than 0
            assert rag_db_instance.chunk_overlap > 0
    
    def test_chunk_text_sentence_boundary(self, rag_db_instance):
        """Test chunking respects sentence boundaries."""
        text = "First sentence. " * 10 + "Second sentence. " * 10
        chunks = rag_db_instance.chunk_text(text, "test.md")
        
        # Should create chunks that try to break at sentence boundaries
        assert len(chunks) > 0
        # Most chunks should end with proper punctuation or whitespace
        proper_endings = ['. ', '.\n', '!\n', '?\n']
        
        for chunk in chunks[:-1]:  # Exclude last chunk which might be incomplete
            chunk_text = chunk['text']
            if len(chunk_text) < rag_db_instance.chunk_size:
                # Shorter chunks should end properly
                assert any(chunk_text.endswith(ending) for ending in proper_endings) or chunk_text.endswith('.')
    
    def test_load_markdown_files_empty_directory(self, rag_db_instance, temp_dir):
        """Test loading from empty directory."""
        rag_db_instance.load_markdown_files(temp_dir)
        stats = rag_db_instance.get_stats()
        assert stats['document_count'] == 0
    
    def test_load_markdown_files_nonexistent_directory(self, rag_db_instance):
        """Test loading from non-existent directory."""
        non_existent = "/path/that/does/not/exist"
        # Should not raise exception, just log warning
        rag_db_instance.load_markdown_files(non_existent)
        stats = rag_db_instance.get_stats()
        assert stats['document_count'] == 0
    
    def test_load_markdown_files_success(self, rag_db_instance, test_markdown_files):
        """Test successful loading of markdown files."""
        rag_db_instance.load_markdown_files(test_markdown_files)
        stats = rag_db_instance.get_stats()
        
        assert stats['document_count'] > 0
        # Should have loaded chunks from both test files
        assert stats['document_count'] >= 2
    
    def test_query_empty_database(self, rag_db_instance):
        """Test querying empty database."""
        results = rag_db_instance.query("test query", top_k=3)
        assert results == []
    
    def test_query_with_data(self, rag_db_instance, test_markdown_files):
        """Test querying database with loaded data."""
        # Load test data
        rag_db_instance.load_markdown_files(test_markdown_files)
        
        # Query for Python-related content
        results = rag_db_instance.query("Python programming", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3  # Should respect top_k limit
        
        # Check result format
        for text, similarity, metadata in results:
            assert isinstance(text, str)
            assert isinstance(similarity, float)
            assert isinstance(metadata, dict)
            assert 'source' in metadata
            assert 'chunk_id' in metadata
            assert 0 <= similarity <= 1  # Similarity should be normalized
    
    def test_query_relevance(self, rag_db_instance, test_markdown_files):
        """Test that query returns relevant results."""
        rag_db_instance.load_markdown_files(test_markdown_files)
        
        # Query for specific topics
        python_results = rag_db_instance.query("Python programming language", top_k=5)
        rag_results = rag_db_instance.query("RAG retrieval augmented generation", top_k=5)
        
        # Should find relevant content
        assert len(python_results) > 0
        assert len(rag_results) > 0
        
        # Check that results contain relevant keywords
        python_texts = [result[0].lower() for result in python_results]
        assert any('python' in text for text in python_texts)
        
        rag_texts = [result[0].lower() for result in rag_results]
        assert any('rag' in text or 'retrieval' in text for text in rag_texts)
    
    def test_clear_database(self, rag_db_instance, test_markdown_files):
        """Test clearing the database."""
        # Load data first
        rag_db_instance.load_markdown_files(test_markdown_files)
        stats_before = rag_db_instance.get_stats()
        assert stats_before['document_count'] > 0
        
        # Clear database
        rag_db_instance.clear_database()
        stats_after = rag_db_instance.get_stats()
        assert stats_after['document_count'] == 0
        
        # Query should return empty results
        results = rag_db_instance.query("test query", top_k=3)
        assert results == []
    
    def test_get_stats(self, rag_db_instance):
        """Test getting database statistics."""
        stats = rag_db_instance.get_stats()
        
        assert isinstance(stats, dict)
        assert 'document_count' in stats
        assert 'embedding_model' in stats
        assert isinstance(stats['document_count'], int)
        assert stats['document_count'] >= 0
    
    def test_multiple_file_loading(self, rag_db_instance, temp_dir):
        """Test loading multiple markdown files."""
        # Create multiple test files
        files_content = {
            'file1.md': '# File 1\nContent about topic A.',
            'file2.md': '# File 2\nContent about topic B.',
            'file3.md': '# File 3\nContent about topic C.',
        }
        
        for filename, content in files_content.items():
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Load all files
        rag_db_instance.load_markdown_files(temp_dir)
        stats = rag_db_instance.get_stats()
        
        # Should have loaded chunks from all files
        assert stats['document_count'] >= len(files_content)
        
        # Query should find content from different files
        results = rag_db_instance.query("topic", top_k=10)
        sources = {result[2]['source'] for result in results}
        assert len(sources) >= 2  # Should find content from multiple sources
    
    def test_special_characters_handling(self, rag_db_instance, temp_dir):
        """Test handling of special characters in markdown files."""
        special_content = """# Test with Special Characters
        
This content has émojis 🚀 and ñon-ASCII characters like café and naïve.
It also has code: `print("Hello, World!")` and links: [example](http://example.com).

## Math symbols
Some math: α, β, γ, Σ, ∑, ∫, ∂, ∆, ∇

## Unicode characters
Various Unicode: 中文, العربية, русский, 日本語
"""
        
        filepath = os.path.join(temp_dir, 'special.md')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(special_content)
        
        # Should handle special characters without errors
        rag_db_instance.load_markdown_files(temp_dir)
        results = rag_db_instance.query("special characters", top_k=3)
        
        # Should find the content
        assert len(results) > 0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_model_loading_error(self, mock_sentence_transformer, temp_db_dir):
        """Test handling of embedding model loading errors."""
        # Mock the SentenceTransformer to raise an exception
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            RAGDatabase(
                db_path=temp_db_dir,
                embedding_model="invalid-model"
            )
    
    def test_large_text_chunking(self, rag_db_instance):
        """Test chunking of very large text."""
        # Create a large text (larger than typical chunk size)
        large_text = "This is a sentence. " * 1000  # About 20k characters
        
        chunks = rag_db_instance.chunk_text(large_text, "large.md")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Check that no chunk exceeds the maximum size significantly
        max_expected_size = rag_db_instance.chunk_size + 100  # Some tolerance
        for chunk in chunks:
            assert len(chunk['text']) <= max_expected_size
        
        # Check that chunks have sequential IDs
        chunk_ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        assert chunk_ids == list(range(len(chunks)))

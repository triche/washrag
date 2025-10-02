"""
Simplified test for RAG Database component.
"""

import os
import sys
import tempfile
import shutil
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRAGDatabaseSimple:
    """Simplified test suite for RAGDatabase class."""
    
    def test_import(self):
        """Test that we can import RAGDatabase."""
        try:
            from rag_database import RAGDatabase
            assert RAGDatabase is not None
        except ImportError:
            pytest.fail("Could not import RAGDatabase")
    
    def test_initialization_simple(self):
        """Test RAGDatabase initialization with minimal setup."""
        from rag_database import RAGDatabase
        
        # Create temporary directory
        temp_db = tempfile.mkdtemp()
        
        try:
            # Initialize with minimal parameters
            rag_db = RAGDatabase(db_path=temp_db)
            
            # Basic assertions
            assert rag_db.db_path == temp_db
            assert hasattr(rag_db, 'chunk_size')
            assert hasattr(rag_db, 'chunk_overlap')
            assert hasattr(rag_db, 'embedding_model')
            assert hasattr(rag_db, 'client')
            assert hasattr(rag_db, 'collection')
            
        finally:
            # Always cleanup
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_chunking_simple(self):
        """Test basic text chunking."""
        from rag_database import RAGDatabase
        
        temp_db = tempfile.mkdtemp()
        
        try:
            rag_db = RAGDatabase(db_path=temp_db, chunk_size=50, chunk_overlap=10)
            
            # Test simple text
            test_text = "This is a test. " * 10  # Create text longer than chunk_size
            chunks = rag_db.chunk_text(test_text, "test.md")
            
            # Basic assertions
            assert len(chunks) > 0
            assert all(isinstance(chunk, dict) for chunk in chunks)
            assert all('text' in chunk for chunk in chunks)
            assert all('metadata' in chunk for chunk in chunks)
            
            # Check metadata
            for chunk in chunks:
                assert chunk['metadata']['source'] == 'test.md'
                assert 'chunk_id' in chunk['metadata']
                
        finally:
            shutil.rmtree(temp_db, ignore_errors=True)

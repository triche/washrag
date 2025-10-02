"""
Performance tests for the WashRAG application.
"""

import os
import sys
import time
import tempfile
import shutil
import pytest
from typing import List

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_database import RAGDatabase # pylint: disable=import-error
from .test_utils import create_sample_knowledge_base


class TestPerformance:
    """Performance tests for WashRAG components."""
    
    @pytest.mark.slow
    def test_large_document_loading_performance(self):
        """Test performance of loading large documents."""
        temp_dir = tempfile.mkdtemp()
        temp_db = tempfile.mkdtemp()
        
        try:
            # Create large documents
            large_content = "This is a test sentence. " * 1000  # ~25KB
            
            # Create multiple large files
            for i in range(10):
                filepath = os.path.join(temp_dir, f'large_doc_{i}.md')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Large Document {i}\n\n{large_content}")
            
            # Measure loading time
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=500,
                chunk_overlap=50
            )
            
            start_time = time.time()
            rag_db.load_markdown_files(temp_dir)
            loading_time = time.time() - start_time
            
            # Performance assertions
            assert loading_time < 60  # Should complete within 60 seconds
            
            stats = rag_db.get_stats()
            assert stats['document_count'] > 50  # Should create many chunks
            
            print(f"Loading time: {loading_time:.2f} seconds")
            print(f"Documents loaded: {stats['document_count']}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_db, ignore_errors=True)
    
    @pytest.mark.slow
    def test_query_performance_with_large_database(self):
        """Test query performance with large database."""
        temp_dir = tempfile.mkdtemp()
        temp_db = tempfile.mkdtemp()
        
        try:
            # Create diverse content for testing
            create_sample_knowledge_base(temp_dir)
            
            # Add more files for larger database
            topics = [
                ("algorithms", "sorting searching optimization"),
                ("databases", "SQL NoSQL transactions indexing"),
                ("networking", "protocols TCP UDP HTTP"),
                ("security", "encryption authentication authorization"),
                ("testing", "unit integration end-to-end")
            ]
            
            for topic, keywords in topics:
                content = f"""# {topic.title()}

This document covers {topic} concepts and practices.

## Overview
{keywords} are important aspects of {topic}.

## Details
{'This section provides detailed information about ' + topic + '. ' * 100}

## Examples
{'Here are practical examples of ' + topic + ' implementation. ' * 50}
"""
                filepath = os.path.join(temp_dir, f'{topic}.md')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Load database
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=300,
                chunk_overlap=30
            )
            rag_db.load_markdown_files(temp_dir)
            
            # Test multiple queries and measure performance
            test_queries = [
                "Python programming",
                "machine learning algorithms",
                "web development frameworks",
                "data science libraries",
                "database optimization",
                "network security",
                "testing strategies"
            ]
            
            query_times = []
            
            for query in test_queries:
                start_time = time.time()
                results = rag_db.query(query, top_k=5)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Verify results
                assert len(results) <= 5
                assert query_time < 2.0  # Each query should be fast
            
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            
            print(f"Average query time: {avg_query_time:.3f} seconds")
            print(f"Maximum query time: {max_query_time:.3f} seconds")
            
            # Performance assertions
            assert avg_query_time < 0.5  # Average should be very fast
            assert max_query_time < 2.0   # No query should be too slow
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_chunking_performance(self):
        """Test text chunking performance."""
        # Create very large text
        large_text = "This is a test sentence with various content. " * 10000  # ~500KB
        
        rag_db = RAGDatabase(
            db_path=tempfile.mkdtemp(),
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Measure chunking time
        start_time = time.time()
        chunks = rag_db.chunk_text(large_text, "large_test.md")
        chunking_time = time.time() - start_time
        
        # Performance and correctness assertions
        assert chunking_time < 5.0  # Should be fast
        assert len(chunks) > 500    # Should create many chunks
        
        # Verify chunk quality
        for chunk in chunks[:10]:  # Check first 10 chunks
            assert len(chunk['text']) <= 600  # Should respect size limits
            assert 'metadata' in chunk
            assert chunk['metadata']['source'] == 'large_test.md'
        
        print(f"Chunking time: {chunking_time:.3f} seconds")
        print(f"Chunks created: {len(chunks)}")
    
    def test_memory_usage_with_large_dataset(self):
        """Test memory usage with large dataset."""
        import psutil
        import os as system_os
        
        # Get initial memory usage
        process = psutil.Process(system_os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        temp_dir = tempfile.mkdtemp()
        temp_db = tempfile.mkdtemp()
        
        try:
            # Create substantial content
            for i in range(20):
                content = f"""# Document {i}

{'This is substantial content for memory testing. ' * 500}

## Section A
{'More content with different patterns and information. ' * 300}

## Section B  
{'Additional detailed content for comprehensive testing. ' * 200}
"""
                filepath = os.path.join(temp_dir, f'memory_test_{i}.md')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Load into database
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=400,
                chunk_overlap=40
            )
            rag_db.load_markdown_files(temp_dir)
            
            # Check memory after loading
            after_loading_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = after_loading_memory - initial_memory
            
            # Perform multiple queries to test memory stability
            for i in range(50):
                rag_db.query(f"test query {i}", top_k=3)
            
            # Check final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            query_memory_increase = final_memory - after_loading_memory
            
            print(f"Initial memory: {initial_memory:.1f} MB")
            print(f"After loading: {after_loading_memory:.1f} MB")
            print(f"Final memory: {final_memory:.1f} MB")
            print(f"Loading increase: {memory_increase:.1f} MB")
            print(f"Query increase: {query_memory_increase:.1f} MB")
            
            # Memory usage assertions (these are rough guidelines)
            assert memory_increase < 500    # Loading shouldn't use excessive memory
            assert query_memory_increase < 50  # Queries shouldn't leak memory significantly
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        import threading
        import concurrent.futures
        
        temp_dir = tempfile.mkdtemp()
        temp_db = tempfile.mkdtemp()
        
        try:
            # Setup database with test data
            create_sample_knowledge_base(temp_dir)
            
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2"
            )
            rag_db.load_markdown_files(temp_dir)
            
            # Define query function
            def perform_queries(query_count: int) -> List[float]:
                times = []
                for i in range(query_count):
                    start_time = time.time()
                    rag_db.query(f"test query {i}", top_k=3)
                    times.append(time.time() - start_time)
                return times
            
            # Test concurrent queries
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(perform_queries, 10) for _ in range(4)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            # Analyze results
            all_times = [time for result in results for time in result]
            avg_time = sum(all_times) / len(all_times)
            max_time = max(all_times)
            
            print(f"Total concurrent test time: {total_time:.2f} seconds")
            print(f"Average query time: {avg_time:.3f} seconds")
            print(f"Maximum query time: {max_time:.3f} seconds")
            print(f"Total queries: {len(all_times)}")
            
            # Performance assertions
            assert total_time < 30      # Should complete reasonably quickly
            assert avg_time < 1.0       # Individual queries should be fast
            assert max_time < 5.0       # No query should be extremely slow
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_db, ignore_errors=True)

"""
Integration tests for the WashRAG application.

These tests verify that different components work together correctly.
"""

import os
import sys
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_database import RAGDatabase # pylint: disable=import-error
from agent import AIAgent # pylint: disable=import-error
import main


class TestIntegration:
    """Integration tests for the WashRAG system."""
    
    def test_end_to_end_workflow(self, config_file, test_markdown_files, mock_openai_client):
        """Test complete end-to-end workflow."""
        with patch('agent.OpenAI', return_value=mock_openai_client):
            with patch.dict(os.environ, {'TEST_OPENAI_API_KEY': 'test-key'}):
                # Initialize agent
                agent = AIAgent(config_file)
                
                # Load knowledge base
                agent.load_knowledge_base(test_markdown_files)
                
                # Verify knowledge base is loaded
                stats = agent.rag_db.get_stats()
                assert stats['document_count'] > 0
                
                # Test querying
                result = agent.chat("What is Python?")
                
                # Verify response structure
                assert 'response' in result
                assert 'sources' in result
                assert 'context_chunks' in result
                
                # Should have found relevant context
                assert result['context_chunks'] > 0
                assert len(result['sources']) > 0
    
    def test_rag_database_with_real_files(self):
        """Test RAG database with actual project files."""
        # Test with the actual knowledge base files
        kb_path = os.path.join(os.path.dirname(__file__), '..', 'rag_db')
        
        if not os.path.exists(kb_path):
            pytest.skip("Real knowledge base files not found")
        
        # Create temporary database
        temp_db = tempfile.mkdtemp()
        
        try:
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=300,
                chunk_overlap=30
            )
            
            # Load actual markdown files
            rag_db.load_markdown_files(kb_path)
            
            # Verify files were loaded
            stats = rag_db.get_stats()
            assert stats['document_count'] > 0
            
            # Test relevant queries
            python_results = rag_db.query("What is WashRAG?", top_k=3)
            assert len(python_results) > 0
            
            # Check that results are relevant
            for text, similarity, metadata in python_results:
                assert similarity > 0.2  # Should be reasonably relevant (adjusted for test client)
                assert 'source' in metadata
                
        finally:
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_agent_with_real_config(self, mock_openai_client):
        """Test agent with actual configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agent_config.yaml')
        
        if not os.path.exists(config_path):
            pytest.skip("Real config file not found")
        
        with patch('agent.OpenAI', return_value=mock_openai_client):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                agent = AIAgent(config_path)
                
                # Verify agent properties from real config
                assert agent.name == "WashRAG Assistant"
                assert agent.temperature == 0.7
                assert agent.max_tokens == 1000
                assert agent.top_k == 3
                assert agent.similarity_threshold == 0.7
    
    def test_cli_argument_parsing_integration(self):
        """Test CLI argument parsing with various combinations."""
        import argparse
        
        # Create parser like in main.py
        parser = argparse.ArgumentParser()
        parser.add_argument('-q', '--query', help='Single query to process')
        parser.add_argument('-c', '--config', help='Path to config file')
        parser.add_argument('--kb-path', help='Path to knowledge base directory')
        parser.add_argument('--no-load', action='store_true', help='Skip loading KB')
        
        # Test various argument combinations
        test_cases = [
            (['-q', 'test query'], {'query': 'test query', 'config': None, 'no_load': False}),
            (['--config', 'test.yaml'], {'query': None, 'config': 'test.yaml', 'no_load': False}),
            (['--no-load', '--kb-path', './docs'], {'no_load': True, 'kb_path': './docs'}),
            (['-q', 'test', '-c', 'config.yaml'], {'query': 'test', 'config': 'config.yaml'}),
        ]
        
        for args_list, expected_attrs in test_cases:
            args = parser.parse_args(args_list)
            for attr, expected_value in expected_attrs.items():
                assert getattr(args, attr) == expected_value
    
    @patch('main.print_banner')
    @patch('main.initialize_agent')
    @patch('main.load_knowledge_base')
    @patch('main.single_query_mode')
    def test_main_function_integration(self, mock_single, mock_load, mock_init, mock_banner):
        """Test main function with mocked components."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_init.return_value = mock_agent
        mock_load.return_value = True
        
        # Test with query argument
        test_args = ['main.py', '-q', 'test query']
        
        with patch('sys.argv', test_args):
            main.main()
        
        # Verify call sequence
        mock_banner.assert_called_once()
        mock_init.assert_called_once()
        mock_load.assert_called_once()
        mock_single.assert_called_once_with(mock_agent, 'test query')
    
    def test_error_handling_integration(self, config_file, test_markdown_files):
        """Test error handling across components."""
        # Test with invalid API key
        with patch.dict(os.environ, {}, clear=True):
            agent = AIAgent(config_file)
            agent.load_knowledge_base(test_markdown_files)
            
            # Should handle missing API key gracefully
            result = agent.chat("test query")
            assert "ERROR: OpenAI API key not configured" in result['response']
    
    def test_performance_with_large_dataset(self):
        """Test system performance with larger dataset."""
        # Create temporary files and database
        temp_dir = tempfile.mkdtemp()
        temp_db = tempfile.mkdtemp()
        
        try:
            # Create multiple markdown files with substantial content
            for i in range(5):
                content = f"""# Document {i}

This is document number {i} with substantial content for performance testing.

## Section A
{'This is content about various topics. ' * 100}

## Section B  
{'More content with different information. ' * 100}

## Section C
{'Additional content for testing purposes. ' * 100}
"""
                filepath = os.path.join(temp_dir, f'doc_{i}.md')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Initialize RAG database
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Load files (this tests chunking performance)
            rag_db.load_markdown_files(temp_dir)
            
            # Verify loading
            stats = rag_db.get_stats()
            assert stats['document_count'] > 10  # Should create multiple chunks
            
            # Test query performance
            results = rag_db.query("testing content information", top_k=5)
            assert len(results) > 0
            
            # All results should have reasonable similarity scores
            for text, similarity, metadata in results:
                assert 0 <= similarity <= 1
                assert len(text) > 0
                assert 'source' in metadata
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_concurrent_database_operations(self):
        """Test concurrent operations on the database."""
        import threading
        import time
        
        temp_db = tempfile.mkdtemp()
        
        try:
            rag_db = RAGDatabase(
                db_path=temp_db,
                embedding_model="all-MiniLM-L6-v2",
                chunk_size=200,
                chunk_overlap=20
            )
            
            # Create test content
            temp_dir = tempfile.mkdtemp()
            test_content = "Test content for concurrent operations. " * 50
            
            test_file = os.path.join(temp_dir, 'test.md')
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Load initial data
            rag_db.load_markdown_files(temp_dir)
            
            # Function to perform queries concurrently
            def query_worker(results_list):
                try:
                    results = rag_db.query("test content", top_k=3)
                    results_list.append(len(results))
                except Exception as e:
                    results_list.append(f"Error: {e}")
            
            # Run multiple queries concurrently
            results_lists = []
            threads = []
            
            for i in range(3):
                results_list = []
                results_lists.append(results_list)
                thread = threading.Thread(target=query_worker, args=(results_list,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Check results
            for results_list in results_lists:
                assert len(results_list) == 1
                assert isinstance(results_list[0], int)  # Should be count, not error
                assert results_list[0] > 0  # Should find results
                
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        finally:
            shutil.rmtree(temp_db, ignore_errors=True)
    
    def test_system_resilience(self, config_file):
        """Test system resilience to various failure scenarios."""
        # Test 1: Database corruption simulation
        temp_db = tempfile.mkdtemp()
        
        try:
            # Create database and load data
            rag_db = RAGDatabase(db_path=temp_db)
            
            # Simulate corruption by removing database files
            shutil.rmtree(temp_db)
            
            # Operations should handle missing database gracefully
            try:
                results = rag_db.query("test", top_k=3)
                # Should either return empty results or handle error gracefully
                assert isinstance(results, list)
            except Exception:
                # Exception is acceptable, should not crash
                pass
                
        finally:
            if os.path.exists(temp_db):
                shutil.rmtree(temp_db, ignore_errors=True)
        
        # Test 2: Network/API failure simulation
        with patch('agent.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Network error")
            mock_openai.return_value = mock_client
            
            with patch.dict(os.environ, {'TEST_OPENAI_API_KEY': 'test-key'}):
                agent = AIAgent(config_file)
                result = agent.chat("test query")
                
                # Should handle API errors gracefully
                assert 'response' in result
                assert "Error generating response" in result['response']

"""
Tests for the AI Agent component.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import AIAgent


class TestAIAgent:
    """Test suite for AIAgent class."""
    
    def test_initialization_with_valid_config(self, config_file, mock_openai_client):
        """Test AIAgent initialization with valid configuration."""
        with patch('agent.OpenAI', return_value=mock_openai_client):
            with patch.dict(os.environ, {'TEST_OPENAI_API_KEY': 'test-key'}):
                agent = AIAgent(config_file)
                
                assert agent.name == 'Test Agent'
                assert agent.personality == 'Helpful test assistant'
                assert agent.temperature == 0.7
                assert agent.max_tokens == 500
                assert agent.top_k == 3
                assert agent.similarity_threshold == 0.7
                assert agent.client is not None
    
    def test_initialization_without_api_key(self, config_file):
        """Test AIAgent initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):  # Clear environment
            agent = AIAgent(config_file)
            
            assert agent.client is None
            assert agent.name == 'Test Agent'  # Other config should still work
    
    def test_initialization_invalid_config(self, temp_dir):
        """Test AIAgent initialization with invalid config file."""
        invalid_config = os.path.join(temp_dir, 'invalid.yaml')
        with open(invalid_config, 'w') as f:
            f.write("invalid: yaml: content:")
        
        with pytest.raises(Exception):
            AIAgent(invalid_config)
    
    def test_initialization_missing_config(self):
        """Test AIAgent initialization with missing config file."""
        with pytest.raises(FileNotFoundError):
            AIAgent("/path/that/does/not/exist.yaml")
    
    @patch('agent.RAGDatabase')
    def test_load_knowledge_base(self, mock_rag_db, agent_instance, test_markdown_files):
        """Test loading knowledge base."""
        # Setup mock
        mock_rag_instance = Mock()
        mock_rag_instance.get_stats.return_value = {'document_count': 5}
        agent_instance.rag_db = mock_rag_instance
        
        # Test loading
        agent_instance.load_knowledge_base(test_markdown_files)
        
        # Verify calls
        mock_rag_instance.load_markdown_files.assert_called_once_with(test_markdown_files)
        mock_rag_instance.get_stats.assert_called_once()
    
    def test_retrieve_context_with_results(self, agent_instance):
        """Test context retrieval with matching results."""
        # Mock RAG database to return test results
        mock_results = [
            ("This is about Python programming", 0.9, {"source": "test1.md"}),
            ("Python is a programming language", 0.8, {"source": "test2.md"}),
            ("Some unrelated content", 0.6, {"source": "test3.md"})  # Below threshold
        ]
        
        agent_instance.rag_db.query = Mock(return_value=mock_results)
        
        # Test retrieval
        texts, sources = agent_instance.retrieve_context("Python programming")
        
        # Should only return results above similarity threshold (0.7)
        assert len(texts) == 2
        assert len(sources) == 2
        assert "This is about Python programming" in texts
        assert "Python is a programming language" in texts
        
        # Check sources
        assert sources[0]['source'] == 'test1.md'
        assert sources[0]['similarity'] == 0.9
        assert sources[1]['source'] == 'test2.md'
        assert sources[1]['similarity'] == 0.8
    
    def test_retrieve_context_no_results(self, agent_instance):
        """Test context retrieval with no matching results."""
        agent_instance.rag_db.query = Mock(return_value=[])
        
        texts, sources = agent_instance.retrieve_context("non-existent topic")
        
        assert texts == []
        assert sources == []
    
    def test_retrieve_context_below_threshold(self, agent_instance):
        """Test context retrieval with results below similarity threshold."""
        mock_results = [
            ("Low similarity content", 0.5, {"source": "test1.md"}),
            ("Another low similarity", 0.3, {"source": "test2.md"})
        ]
        
        agent_instance.rag_db.query = Mock(return_value=mock_results)
        
        texts, sources = agent_instance.retrieve_context("test query")
        
        # Should return empty as all results are below threshold (0.7)
        assert texts == []
        assert sources == []
    
    def test_generate_response_with_context(self, agent_instance):
        """Test response generation with context."""
        query = "What is Python?"
        context = ["Python is a programming language", "Python is easy to learn"]
        sources = [
            {"source": "python_guide.md", "similarity": 0.9},
            {"source": "programming.md", "similarity": 0.8}
        ]
        
        response = agent_instance.generate_response(query, context, sources)
        
        # Should call OpenAI API
        agent_instance.client.chat.completions.create.assert_called_once()
        
        # Check the call arguments
        call_args = agent_instance.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 2  # system and user messages
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        
        # User message should contain query and context
        user_content = messages[1]['content']
        assert query in user_content
        assert "Python is a programming language" in user_content
        assert "python_guide.md" in user_content
        
        # Should return the mocked response
        assert response == "This is a test response from the AI."
    
    def test_generate_response_without_context(self, agent_instance):
        """Test response generation without context."""
        query = "What is Python?"
        context = []
        sources = []
        
        response = agent_instance.generate_response(query, context, sources)
        
        # Should still call OpenAI API
        agent_instance.client.chat.completions.create.assert_called_once()
        
        # Check user message contains "no relevant information" text
        call_args = agent_instance.client.chat.completions.create.call_args
        user_content = call_args[1]['messages'][1]['content']
        assert "No relevant information found" in user_content
    
    def test_generate_response_without_client(self, config_file):
        """Test response generation without OpenAI client."""
        with patch.dict(os.environ, {}, clear=True):  # No API key
            agent = AIAgent(config_file)
            
            response = agent.generate_response("test query", [], [])
            
            assert "ERROR: OpenAI API key not configured" in response
    
    def test_generate_response_api_error(self, agent_instance):
        """Test response generation with API error."""
        # Mock OpenAI client to raise exception
        agent_instance.client.chat.completions.create.side_effect = Exception("API Error")
        
        response = agent_instance.generate_response("test query", [], [])
        
        assert "Error generating response: API Error" in response
    
    def test_chat_full_workflow(self, agent_instance):
        """Test the complete chat workflow."""
        # Mock RAG database
        mock_results = [
            ("Python is a programming language", 0.9, {"source": "python.md"})
        ]
        agent_instance.rag_db.query = Mock(return_value=mock_results)
        
        # Test chat
        result = agent_instance.chat("What is Python?")
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'sources' in result
        assert 'context_chunks' in result
        
        # Check values
        assert result['response'] == "This is a test response from the AI."
        assert len(result['sources']) == 1
        assert result['sources'][0]['source'] == 'python.md'
        assert result['context_chunks'] == 1
        
        # Verify RAG database was queried
        agent_instance.rag_db.query.assert_called_once_with("What is Python?", top_k=3)
    
    def test_chat_no_context_found(self, agent_instance):
        """Test chat when no relevant context is found."""
        # Mock RAG database to return empty results
        agent_instance.rag_db.query = Mock(return_value=[])
        
        result = agent_instance.chat("Unknown topic")
        
        assert result['response'] == "This is a test response from the AI."
        assert result['sources'] == []
        assert result['context_chunks'] == 0
    
    def test_clear_knowledge_base(self, agent_instance):
        """Test clearing the knowledge base."""
        agent_instance.rag_db.clear_database = Mock()
        
        agent_instance.clear_knowledge_base()
        
        agent_instance.rag_db.clear_database.assert_called_once()
    
    def test_config_loading_missing_sections(self, temp_dir):
        """Test configuration loading with missing required sections."""
        # Create config with missing sections
        incomplete_config = {
            'agent': {
                'name': 'Test Agent'
                # Missing other required fields
            }
            # Missing 'rag' and 'llm' sections
        }
        
        import yaml
        config_path = os.path.join(temp_dir, 'incomplete.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        # Should raise KeyError for missing configuration
        with pytest.raises(KeyError):
            AIAgent(config_path)
    
    def test_temperature_and_max_tokens_configuration(self, agent_instance):
        """Test that temperature and max_tokens are properly configured."""
        query = "test query"
        context = ["test context"]
        sources = [{"source": "test.md", "similarity": 0.9}]
        
        agent_instance.generate_response(query, context, sources)
        
        # Check that the API call used correct parameters
        call_args = agent_instance.client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 500
        assert call_args[1]['model'] == 'gpt-3.5-turbo'
    
    def test_system_prompt_and_personality_inclusion(self, agent_instance):
        """Test that system prompt and personality are included in API calls."""
        agent_instance.generate_response("test query", [], [])
        
        call_args = agent_instance.client.chat.completions.create.call_args
        system_message = call_args[1]['messages'][0]['content']
        
        assert "You are a test AI assistant." in system_message
        assert "Helpful test assistant" in system_message
    
    @patch('agent.logger')
    def test_logging_functionality(self, mock_logger, agent_instance):
        """Test that logging works correctly."""
        # Test chat logging
        agent_instance.rag_db.query = Mock(return_value=[])
        agent_instance.chat("test query")
        
        # Should log the query processing
        mock_logger.info.assert_called()
        
        # Check that query and retrieval info are logged
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Processing query: test query" in call for call in log_calls)
        assert any("Retrieved 0 relevant chunks" in call for call in log_calls)
    
    def test_source_deduplication_in_response(self, agent_instance):
        """Test that duplicate sources are handled properly."""
        query = "test query"
        context = ["Content 1", "Content 2", "Content 3"]
        sources = [
            {"source": "same_file.md", "similarity": 0.9},
            {"source": "same_file.md", "similarity": 0.8},  # Duplicate source
            {"source": "different_file.md", "similarity": 0.7}
        ]
        
        agent_instance.generate_response(query, context, sources)
        
        call_args = agent_instance.client.chat.completions.create.call_args
        user_content = call_args[1]['messages'][1]['content']
        
        # Should list unique sources
        assert "same_file.md, different_file.md" in user_content or "different_file.md, same_file.md" in user_content

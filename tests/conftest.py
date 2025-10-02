"""
Conftest file for pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules only when needed to avoid import errors during test discovery


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Test Document

This is a test document for the RAG system.

## Section 1: Python Programming

Python is a high-level programming language known for its simplicity and readability.
It's widely used in data science, web development, and artificial intelligence.

Key features of Python:
- Easy to learn and use
- Extensive standard library
- Large community support
- Cross-platform compatibility

## Section 2: RAG Systems

RAG stands for Retrieval Augmented Generation.
It's a technique that combines information retrieval with text generation.

RAG systems work by:
1. Storing documents in a vector database
2. Retrieving relevant information based on queries
3. Using retrieved context to generate responses

## Section 3: Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.

Common ML algorithms include:
- Linear regression
- Decision trees
- Neural networks
- Support vector machines
"""


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'agent': {
            'name': 'Test Agent',
            'personality': 'Helpful test assistant',
            'temperature': 0.7,
            'max_tokens': 500,
            'system_prompt': 'You are a test AI assistant.'
        },
        'rag': {
            'chunk_size': 200,
            'chunk_overlap': 20,
            'top_k': 3,
            'similarity_threshold': 0.7,
            'embedding_model': 'all-MiniLM-L6-v2'
        },
        'llm': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'api_key_env': 'TEST_OPENAI_API_KEY'
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Create a temporary config file."""
    import yaml
    
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f)
    
    return config_path


@pytest.fixture
def test_markdown_files(temp_dir, sample_markdown_content):
    """Create test markdown files in a temporary directory."""
    # Create main test file
    test_file1 = os.path.join(temp_dir, 'test_doc1.md')
    with open(test_file1, 'w', encoding='utf-8') as f:
        f.write(sample_markdown_content)
    
    # Create second test file
    test_content2 = """# Another Test Document

This document contains additional information for testing.

## Web Development

Web development involves creating websites and web applications.
Popular frameworks include Django, Flask, and FastAPI for Python.

## Data Science

Data science combines statistics, programming, and domain expertise.
Common tools include pandas, numpy, and scikit-learn.
"""
    
    test_file2 = os.path.join(temp_dir, 'test_doc2.md')
    with open(test_file2, 'w', encoding='utf-8') as f:
        f.write(test_content2)
    
    return temp_dir


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    mock_message.content = "This is a test response from the AI."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def rag_db_instance(temp_db_dir):
    """Create a RAGDatabase instance for testing."""
    from rag_database import RAGDatabase  # pylint: disable=import-error
    return RAGDatabase(
        db_path=temp_db_dir,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=20
    )


@pytest.fixture
def agent_instance(config_file, mock_openai_client):
    """Create an AIAgent instance for testing."""
    from agent import AIAgent  # pylint: disable=import-error
    with patch('agent.OpenAI', return_value=mock_openai_client):
        with patch.dict(os.environ, {'TEST_OPENAI_API_KEY': 'test-key'}):
            agent = AIAgent(config_file)
            agent.client = mock_openai_client
            return agent

"""
Test utilities and helper functions.
"""

import os
import tempfile
import shutil
from typing import Dict, Any, List
from unittest.mock import Mock


def create_test_markdown_file(content: str, filename: str = "test.md") -> str:
    """
    Create a temporary markdown file with the given content.
    
    Args:
        content: Markdown content to write
        filename: Name of the file
        
    Returns:
        Path to the created file
    """
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def create_test_config(config_dict: Dict[str, Any]) -> str:
    """
    Create a temporary YAML config file.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Path to the created config file
    """
    import yaml
    
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    return config_path


def cleanup_temp_paths(*paths: str) -> None:
    """
    Clean up temporary paths.
    
    Args:
        *paths: Paths to clean up
    """
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)


def mock_openai_response(content: str = "Test response") -> Mock:
    """
    Create a mock OpenAI response.
    
    Args:
        content: Response content
        
    Returns:
        Mock OpenAI client
    """
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


def create_sample_knowledge_base(base_dir: str) -> List[str]:
    """
    Create a sample knowledge base with multiple markdown files.
    
    Args:
        base_dir: Base directory to create files in
        
    Returns:
        List of created file paths
    """
    files_content = {
        'python_basics.md': """# Python Basics

Python is a high-level programming language known for its simplicity and readability.

## Key Features
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community support

## Getting Started
To start programming in Python, you need to install the Python interpreter.
""",
        
        'web_development.md': """# Web Development with Python

Python offers several frameworks for web development.

## Popular Frameworks
- **Django**: Full-featured web framework
- **Flask**: Lightweight and flexible
- **FastAPI**: Modern, fast web framework

## Best Practices
- Use virtual environments
- Follow PEP 8 style guide
- Write tests for your code
""",
        
        'data_science.md': """# Data Science with Python

Python is widely used in data science and machine learning.

## Essential Libraries
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning

## Common Tasks
- Data cleaning and preprocessing
- Exploratory data analysis
- Model building and evaluation
""",
        
        'machine_learning.md': """# Machine Learning

Machine learning enables computers to learn without being explicitly programmed.

## Types of ML
1. **Supervised Learning**: Learning with labeled data
2. **Unsupervised Learning**: Finding patterns in unlabeled data
3. **Reinforcement Learning**: Learning through interaction

## Algorithms
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
"""
    }
    
    created_files = []
    for filename, content in files_content.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(filepath)
    
    return created_files


def assert_valid_rag_result(result: tuple) -> None:
    """
    Assert that a RAG query result has the correct format.
    
    Args:
        result: Tuple of (text, similarity, metadata)
    """
    assert len(result) == 3
    text, similarity, metadata = result
    
    assert isinstance(text, str)
    assert len(text) > 0
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    
    assert isinstance(metadata, dict)
    assert 'source' in metadata
    assert 'chunk_id' in metadata


def assert_valid_chat_result(result: Dict[str, Any]) -> None:
    """
    Assert that a chat result has the correct format.
    
    Args:
        result: Chat result dictionary
    """
    required_keys = ['response', 'sources', 'context_chunks']
    for key in required_keys:
        assert key in result
    
    assert isinstance(result['response'], str)
    assert isinstance(result['sources'], list)
    assert isinstance(result['context_chunks'], int)
    assert result['context_chunks'] >= 0
    
    # Check sources format
    for source in result['sources']:
        assert isinstance(source, dict)
        assert 'source' in source
        assert 'similarity' in source


class TempEnvironment:
    """Context manager for temporarily modifying environment variables."""
    
    def __init__(self, **env_vars):
        self.env_vars = env_vars
        self.original_env = {}
    
    def __enter__(self):
        # Save original values
        for key in self.env_vars:
            self.original_env[key] = os.environ.get(key)
        
        # Set new values
        for key, value in self.env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

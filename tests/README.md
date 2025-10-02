# WashRAG Test Suite

This directory contains a comprehensive test suite for the WashRAG application, built using pytest. The test suite covers all major components and includes unit tests, integration tests, and performance tests.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── test_rag_database.py     # Tests for RAG database component
├── test_agent.py            # Tests for AI agent component  
├── test_main.py             # Tests for CLI application
├── test_integration.py      # Integration tests
├── test_performance.py      # Performance and load tests
└── test_utils.py            # Test utilities and helpers
```

## Test Categories

### Unit Tests
- **test_rag_database.py**: Tests for the RAGDatabase class including:
  - Text chunking functionality
  - Markdown file loading
  - Vector database operations
  - Query functionality
  - Error handling

- **test_agent.py**: Tests for the AIAgent class including:
  - Agent initialization
  - Configuration loading
  - Context retrieval
  - Response generation
  - Chat workflow

- **test_main.py**: Tests for the CLI application including:
  - Argument parsing
  - Interactive mode
  - Single query mode
  - Error handling

### Integration Tests
- **test_integration.py**: End-to-end integration tests including:
  - Complete workflow testing
  - Component interaction verification
  - Real configuration file testing
  - Error handling across components

### Performance Tests
- **test_performance.py**: Performance and scalability tests including:
  - Large document loading
  - Query performance with large datasets
  - Memory usage monitoring
  - Concurrent access testing

## Running Tests

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock pytest-asyncio
```

### Quick Start

Run the test suite using the provided test runner:

```bash
# Run quick tests (unit + integration, no performance)
python run_tests.py

# Run all tests including performance tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run with coverage report
python run_tests.py coverage
```

### Using pytest directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest

# Run with verbose output and coverage
pytest -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_rag_database.py

# Run tests matching a pattern
pytest -k "test_chunking"

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

## Test Configuration

The test suite is configured through `pytest.ini`:

- Test discovery patterns
- Coverage settings
- Test markers for categorization
- Output formatting options

### Test Markers

- `slow`: Marks tests that take longer to run (performance tests)
- `integration`: Marks integration tests
- `unit`: Marks unit tests
- `requires_api_key`: Marks tests that require an OpenAI API key

## Fixtures and Test Utilities

### Key Fixtures (conftest.py)

- `temp_dir`: Provides a temporary directory for test files
- `temp_db_dir`: Provides a temporary directory for database testing
- `sample_markdown_content`: Sample markdown content for testing
- `sample_config`: Sample configuration dictionary
- `config_file`: Temporary configuration file
- `test_markdown_files`: Creates test markdown files
- `mock_openai_client`: Mocked OpenAI client for testing
- `rag_db_instance`: Pre-configured RAGDatabase instance
- `agent_instance`: Pre-configured AIAgent instance

### Test Utilities (test_utils.py)

Helper functions for:
- Creating test files and configurations
- Mocking external dependencies
- Asserting result formats
- Managing temporary environments

## Test Coverage

The test suite aims for high code coverage across all components:

- **RAG Database**: >95% coverage
- **AI Agent**: >90% coverage  
- **CLI Application**: >85% coverage
- **Overall**: >80% coverage

Generate coverage reports:

```bash
# Terminal coverage report
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html:htmlcov
# View at htmlcov/index.html
```

## Mock Strategy

The test suite uses comprehensive mocking to:

- **External APIs**: Mock OpenAI API calls to avoid costs and network dependency
- **File System**: Use temporary directories for safe testing
- **Environment Variables**: Mock environment variables for configuration testing
- **Time-dependent Operations**: Mock time functions for consistent testing

## Performance Testing

Performance tests are marked with `@pytest.mark.slow` and include:

- **Load Testing**: Tests with large document sets
- **Memory Testing**: Memory usage monitoring during operations
- **Concurrency Testing**: Multi-threaded access patterns
- **Scalability Testing**: Performance with increasing data sizes

Run performance tests:

```bash
python run_tests.py performance
# or
pytest -m slow
```

## Continuous Integration

The test suite is designed to work in CI environments:

- All external dependencies are mocked
- Temporary files are properly cleaned up
- Tests are deterministic and repeatable
- Performance tests can be skipped in CI with `-m "not slow"`

## Adding New Tests

When adding new tests:

1. Follow the existing naming conventions (`test_*.py`)
2. Use appropriate fixtures from `conftest.py`
3. Add proper test markers for categorization
4. Mock external dependencies appropriately
5. Include both positive and negative test cases
6. Add integration tests for new features

### Example Test Structure

```python
class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic functionality."""
        # Arrange
        # Act  
        # Assert
        pass
    
    def test_error_handling(self, fixture_name):
        """Test error handling."""
        with pytest.raises(ExpectedException):
            # Test code that should raise exception
            pass
    
    @pytest.mark.slow
    def test_performance(self, fixture_name):
        """Test performance characteristics."""
        # Performance test code
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the `src` directory is in the Python path
2. **Missing Dependencies**: Install all required packages including test dependencies
3. **API Key Warnings**: Mock client is used by default, no real API key needed
4. **Slow Tests**: Use `-m "not slow"` to skip performance tests
5. **Coverage Issues**: Ensure all source files are in the `src` directory

### Debug Mode

Run tests with extra debugging:

```bash
pytest -v -s --tb=long
```

### Specific Test Debugging

```bash
# Run single test with full output
pytest tests/test_rag_database.py::TestRAGDatabase::test_initialization -v -s
```

## Contributing

When contributing tests:

1. Ensure all tests pass locally
2. Add tests for new functionality
3. Maintain or improve coverage percentages
4. Follow the existing code style
5. Update documentation as needed

The test suite is a critical part of maintaining code quality and ensuring the WashRAG application works correctly across different environments and use cases.

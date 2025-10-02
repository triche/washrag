# WashRAG Quick Start Guide

## Overview

WashRAG is a Retrieval Augmented Generation (RAG) system that allows you to create an AI assistant with a custom knowledge base. The system indexes markdown files and uses them to provide informed answers to user queries.

## Quick Start (5 minutes)

### 1. Set up the environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Add your knowledge

Add markdown files to the `rag_db/` directory. For example:

```bash
echo "# My Topic

Information about my topic here." > rag_db/my_topic.md
```

### 3. Run the agent

```bash
# Interactive mode
python main.py

# Single query mode
python main.py -q "What is WashRAG?"
```

## System Architecture

```
User Query → RAG Database → Retrieve Relevant Chunks → LLM → Response
                ↑
         Markdown Files
         (Knowledge Base)
```

## Components

### 1. Knowledge Base (`rag_db/`)
- Store your information as markdown files
- Files are automatically chunked and indexed
- Supports hierarchical organization with headers

### 2. Configuration (`config/agent_config.yaml`)
- **Agent settings**: Personality, name, temperature
- **RAG settings**: Chunk size, similarity threshold, embedding model
- **LLM settings**: Model selection, API configuration

### 3. Core Modules

- **`src/rag_database.py`**: Manages document indexing and retrieval
  - Chunks documents intelligently
  - Creates vector embeddings
  - Performs similarity search
  
- **`src/agent.py`**: Main AI agent
  - Loads configuration
  - Coordinates RAG and LLM
  - Generates responses

- **`main.py`**: CLI interface
  - Interactive chat mode
  - Single query mode
  - Rich terminal output

## Configuration Options

### Agent Personality

Edit `config/agent_config.yaml` to customize:

```yaml
agent:
  name: "Your Agent Name"
  personality: |
    Your agent's personality description here.
  temperature: 0.7  # 0.0 = focused, 1.0 = creative
```

### RAG Parameters

Tune retrieval performance:

```yaml
rag:
  chunk_size: 500        # Characters per chunk
  chunk_overlap: 50      # Overlap for context
  top_k: 3              # Results to retrieve
  similarity_threshold: 0.7  # Minimum relevance
```

### LLM Settings

Change the underlying model:

```yaml
llm:
  model: "gpt-3.5-turbo"  # or "gpt-4", "gpt-4-turbo", etc.
```

## Advanced Usage

### Custom Knowledge Base Location

```bash
python main.py --kb-path /path/to/your/docs
```

### Skip Initial Loading

Useful if you want to load the knowledge base manually:

```bash
python main.py --no-load
```

### Programmatic Usage

```python
from src.agent import AIAgent

# Initialize agent
agent = AIAgent("./config/agent_config.yaml")

# Load knowledge base
agent.load_knowledge_base("./rag_db")

# Query the agent
result = agent.chat("What is Python?")
print(result['response'])
print(f"Sources: {result['sources']}")
```

## Tips for Best Results

### 1. Structure Your Knowledge Base

Use clear headers and organization:

```markdown
# Main Topic

## Subtopic 1

Detailed information here.

## Subtopic 2

More information here.
```

### 2. Chunk Size

- **Small chunks (200-300)**: Better precision, but may lose context
- **Large chunks (800-1000)**: Better context, but less precise retrieval
- **Medium chunks (500)**: Good balance (default)

### 3. Document Quality

- Use clear, concise language
- Include relevant keywords naturally
- Structure information logically
- Update documents regularly

### 4. Similarity Threshold

- **High (0.8+)**: Only very relevant matches
- **Medium (0.6-0.8)**: Balanced relevance (recommended)
- **Low (0.4-0.6)**: More matches, but may include less relevant content

## Troubleshooting

### API Key Issues

```
Error: OpenAI API key not configured
```

**Solution**: Make sure `.env` file exists with valid `OPENAI_API_KEY`

### No Results from Knowledge Base

```
No relevant information found in knowledge base
```

**Possible causes**:
1. Knowledge base is empty or not loaded
2. Query doesn't match any document content
3. Similarity threshold is too high

**Solutions**:
- Add more documents to `rag_db/`
- Use more descriptive query terms
- Lower `similarity_threshold` in config

### Internet Connection Required

The first time you run the system, it needs to download the embedding model (~90MB). After that, it works offline (except for LLM API calls).

## Best Practices

1. **Version Control**: Use git for your knowledge base
2. **Regular Updates**: Keep documents current
3. **Test Queries**: Verify the system can answer your key questions
4. **Monitor Sources**: Check which documents are being used
5. **Iterate**: Refine based on response quality

## Next Steps

- Add more documents to your knowledge base
- Customize the agent personality
- Experiment with different LLM models
- Adjust RAG parameters for your use case
- Build domain-specific assistants

## Getting Help

- Check the main README.md
- Review configuration in `config/agent_config.yaml`
- Run validation: `python validate.py`
- Inspect example documents in `rag_db/`

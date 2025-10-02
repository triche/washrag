# washrag
Basic RAG (Retrieval Augmented Generation) system in Python

WashRAG is a text-based AI agent that uses a RAG database of markdown files to provide informed responses to user queries. The system combines semantic search with large language models to create an intelligent assistant that can reference your custom knowledge base.

## Features

- 📚 **Markdown-based Knowledge Base**: Store information in easy-to-edit markdown files
- 🔍 **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- 🤖 **Configurable AI Agent**: YAML-based configuration for system prompts and personality
- 💬 **Interactive CLI**: Rich terminal interface for chatting with the agent
- 🎯 **Source Attribution**: Responses include references to source documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/triche/washrag.git
cd washrag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Interactive Mode

Start the agent in interactive mode:
```bash
python main.py
```

This will:
1. Load the configuration from `config/agent_config.yaml`
2. Index all markdown files in `rag_db/`
3. Start an interactive chat session

### Single Query Mode

Ask a single question:
```bash
python main.py -q "What is WashRAG?"
```

### Custom Knowledge Base

Use a different directory for your knowledge base:
```bash
python main.py --kb-path /path/to/your/docs
```

## Configuration

The agent is configured via `config/agent_config.yaml`:

### Agent Settings
- **name**: The agent's display name
- **personality**: Description of the agent's personality and behavior
- **system_prompt**: Core instructions for the agent
- **temperature**: LLM temperature (0.0-1.0, higher = more creative)
- **max_tokens**: Maximum response length

### RAG Settings
- **chunk_size**: Size of document chunks in characters
- **chunk_overlap**: Overlap between chunks
- **top_k**: Number of relevant chunks to retrieve
- **similarity_threshold**: Minimum similarity score for relevance
- **embedding_model**: Sentence transformer model to use

### LLM Settings
- **provider**: LLM provider (currently "openai")
- **model**: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
- **api_key_env**: Environment variable containing the API key

## Knowledge Base

Add markdown files to the `rag_db/` directory. The agent will automatically:
1. Load and parse all `.md` files
2. Split them into chunks
3. Generate embeddings
4. Store them in a vector database (ChromaDB)

Example structure:
```
rag_db/
├── about_washrag.md
├── python_best_practices.md
└── your_custom_docs.md
```

## Project Structure

```
washrag/
├── config/
│   └── agent_config.yaml    # Agent configuration
├── rag_db/                   # Knowledge base (markdown files)
│   ├── about_washrag.md
│   └── python_best_practices.md
├── src/
│   ├── agent.py             # Main AI agent
│   └── rag_database.py      # RAG database manager
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment file
└── README.md                # This file
```

## How It Works

1. **Document Ingestion**: Markdown files are loaded and split into overlapping chunks
2. **Embedding**: Each chunk is converted to a vector embedding using a sentence transformer
3. **Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Query Processing**: User queries are embedded and matched against stored chunks
5. **Response Generation**: Relevant chunks are provided as context to the LLM
6. **Answer**: The LLM generates an informed response based on the retrieved context

## CLI Commands

While in interactive mode:
- `/help` - Show available commands
- `/clear` - Clear the knowledge base
- `/quit` or `/exit` - Exit the program

## Requirements

- Python 3.8+
- OpenAI API key
- ~500MB disk space for models and data

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.


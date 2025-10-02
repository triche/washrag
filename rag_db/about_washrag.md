# About WashRAG

WashRAG is a basic RAG (Retrieval Augmented Generation) system built in Python. It allows you to create an AI assistant that can answer questions based on a knowledge base of markdown documents.

## Key Features

- **Document-based Knowledge**: Store information in markdown files that the AI can reference
- **Semantic Search**: Uses embeddings to find relevant information based on meaning, not just keywords
- **Configurable**: Easy-to-edit YAML configuration for system prompt and personality
- **Extensible**: Built on popular libraries like ChromaDB and OpenAI

## How It Works

1. Markdown files are loaded from the `rag_db` directory
2. Documents are split into chunks and embedded using sentence transformers
3. Chunks are stored in a ChromaDB vector database
4. When you ask a question, relevant chunks are retrieved
5. The AI uses those chunks to generate an informed response

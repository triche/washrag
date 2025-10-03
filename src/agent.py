"""
AI Agent

Main agent that uses RAG to answer questions.
"""

import os
import yaml
from typing import Dict, List
import logging
from openai import OpenAI
from dotenv import load_dotenv

from rag_database import RAGDatabase  # pylint: disable=import-error

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAgent:
    """AI Agent with RAG capabilities."""

    def __init__(self, config_path: str = "./config/agent_config.yaml"):
        """
        Initialize the AI agent.

        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Initialize RAG database
        rag_config = self.config["rag"]
        self.rag_db = RAGDatabase(
            db_path="./chroma_db",
            embedding_model=rag_config["embedding_model"],
            chunk_size=rag_config["chunk_size"],
            chunk_overlap=rag_config["chunk_overlap"],
        )

        # Initialize OpenAI client
        llm_config = self.config["llm"]
        api_key = os.getenv(llm_config["api_key_env"])
        if not api_key:
            logger.warning(
                "API key not found in environment variable: %s", llm_config["api_key_env"]
            )
            logger.warning(
                "Agent will not be able to generate responses without an API key."
            )
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = llm_config["model"]

        # Agent settings
        agent_config = self.config["agent"]
        self.name = agent_config["name"]
        self.personality = agent_config["personality"]
        self.system_prompt = agent_config["system_prompt"]
        self.temperature = agent_config["temperature"]
        self.max_tokens = agent_config["max_tokens"]
        self.verbosity = agent_config.get("verbosity", "normal")

        # RAG settings
        self.top_k = rag_config["top_k"]
        self.similarity_threshold = rag_config["similarity_threshold"]

        # Set logging level based on verbosity
        if self.verbosity == "quiet":
            logger.setLevel(logging.ERROR)
        elif self.verbosity == "verbose":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        if self.verbosity == "verbose":
            logger.info("Agent '%s' initialized", self.name)

    def load_knowledge_base(self, directory: str = "./rag_db"):
        """
        Load markdown files into the knowledge base.

        Args:
            directory: Directory containing markdown files
        """
        if self.verbosity == "verbose":
            logger.info("Loading knowledge base from %s", directory)
        self.rag_db.load_markdown_files(directory)
        stats = self.rag_db.get_stats()
        if self.verbosity == "verbose":
            logger.info("Knowledge base loaded: %s chunks", stats["document_count"])

    def retrieve_context(self, query: str) -> tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context from the knowledge base.

        Args:
            query: User query

        Returns:
            Tuple of (relevant_texts, sources)
        """
        results = self.rag_db.query(query, top_k=self.top_k)

        relevant_texts = []
        sources = []

        for text, similarity, metadata in results:
            if similarity >= self.similarity_threshold:
                relevant_texts.append(text)
                sources.append({"source": metadata["source"], "similarity": similarity})

        return relevant_texts, sources

    def generate_response(
        self, query: str, context: List[str], sources: List[Dict]
    ) -> str:
        """
        Generate a response using the LLM with retrieved context.

        Args:
            query: User query
            context: Retrieved context texts
            sources: Source information for the context

        Returns:
            Generated response
        """
        if not self.client:
            return (
                "ERROR: OpenAI API key not configured. Please set the OPENAI_API_KEY "
                "environment variable to use the agent."
            )

        # Build context string
        if context:
            context_str = "\n\n---\n\n".join(context)
            context_section = (
                f"\n\nRelevant information from knowledge base:\n{context_str}"
            )

            # Add source information
            source_files = list(set(s["source"] for s in sources))
            sources_str = ", ".join(source_files)
            context_section += f"\n\n(Sources: {sources_str})"
        else:
            context_section = "\n\nNo relevant information found in knowledge base."

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt + "\n\nPersonality:\n" + self.personality,
            },
            {"role": "user", "content": query + context_section},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except (ConnectionError, TimeoutError) as e:
            logger.error("Network error generating response: %s", e)
            return f"Network error generating response: {e}"
        except ValueError as e:
            logger.error("Invalid response from API: %s", e)
            return f"Invalid response from API: {e}"
        except RuntimeError as e:
            logger.error("Runtime error generating response: %s", e)
            return f"Runtime error generating response: {e}"

    def chat(self, query: str) -> Dict:
        """
        Process a query and return a response.

        Args:
            query: User query

        Returns:
            Dictionary with response and metadata
        """
        if self.verbosity == "verbose":
            logger.info("Processing query: %s", query)

        # Retrieve relevant context
        context, sources = self.retrieve_context(query)

        if self.verbosity == "verbose":
            logger.info("Retrieved %s relevant chunks", len(context))

        # Generate response
        response = self.generate_response(query, context, sources)

        return {"response": response, "sources": sources, "context_chunks": len(context)}

    def clear_knowledge_base(self):
        """Clear the knowledge base."""
        self.rag_db.clear_database()
        if self.verbosity == "verbose":
            logger.info("Knowledge base cleared")

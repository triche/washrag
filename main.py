#!/usr/bin/env python3
"""
WashRAG CLI

Command-line interface for the WashRAG AI agent.
"""

import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import AIAgent  # pylint: disable=import-error

console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
    ╦ ╦┌─┐┌─┐┬ ┬╦═╗╔═╗╔═╗
    ║║║├─┤└─┐├─┤╠╦╝╠═╣║ ╦
    ╚╩╝┴ ┴└─┘┴ ┴╩╚═╩ ╩╚═╝
    Basic RAG System
    """
    console.print(Panel(banner, style="bold blue"))


def initialize_agent(args):
    """Initialize the AI agent."""
    config_path = args.config or "./config/agent_config.yaml"
    
    if not os.path.exists(config_path):
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        sys.exit(1)
    
    try:
        agent = AIAgent(config_path)
        return agent
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        sys.exit(1)


def load_knowledge_base(agent, args):
    """Load the knowledge base."""
    kb_path = args.kb_path or "./rag_db"
    
    if not os.path.exists(kb_path):
        console.print(f"[yellow]Warning: Knowledge base directory not found: {kb_path}[/yellow]")
        console.print("[yellow]Creating directory...[/yellow]")
        os.makedirs(kb_path, exist_ok=True)
        console.print("[yellow]Please add markdown files to this directory.[/yellow]")
        return False
    
    try:
        with console.status("[bold green]Loading knowledge base..."):
            agent.load_knowledge_base(kb_path)
        console.print("[green]✓ Knowledge base loaded successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error loading knowledge base: {e}[/red]")
        return False


def interactive_mode(agent):
    """Run the agent in interactive mode."""
    console.print("\n[bold green]Interactive Mode[/bold green]")
    console.print("Type your questions. Commands: /help, /clear, /quit\n")
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if not query.strip():
                continue
            
            # Handle commands
            if query.lower() == '/quit' or query.lower() == '/exit':
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            elif query.lower() == '/help':
                console.print(Panel(
                    "[bold]Available Commands:[/bold]\n"
                    "/help  - Show this help message\n"
                    "/clear - Clear the knowledge base\n"
                    "/quit  - Exit the program",
                    title="Help"
                ))
                continue
            
            elif query.lower() == '/clear':
                agent.clear_knowledge_base()
                console.print("[yellow]Knowledge base cleared[/yellow]")
                continue
            
            # Process query
            with console.status("[bold green]Thinking..."):
                result = agent.chat(query)
            
            # Display response
            console.print(f"\n[bold magenta]{agent.name}[/bold magenta]:")
            console.print(Markdown(result['response']))
            
            # Display sources if any
            if result['sources']:
                sources_str = ", ".join(set(s['source'] for s in result['sources']))
                console.print(f"\n[dim]Sources: {sources_str}[/dim]")
            
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def single_query_mode(agent, query):
    """Process a single query and exit."""
    try:
        with console.status("[bold green]Processing query..."):
            result = agent.chat(query)
        
        console.print("\n[bold magenta]Response:[/bold magenta]")
        console.print(Markdown(result['response']))
        
        if result['sources']:
            sources_str = ", ".join(set(s['source'] for s in result['sources']))
            console.print(f"\n[dim]Sources: {sources_str}[/dim]")
        
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WashRAG - Basic RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode
  %(prog)s -q "What is WashRAG?"     # Single query
  %(prog)s --kb-path ./docs          # Use custom knowledge base
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        help='Single query to process (non-interactive mode)'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Path to config file (default: ./config/agent_config.yaml)'
    )
    
    parser.add_argument(
        '--kb-path',
        help='Path to knowledge base directory (default: ./rag_db)'
    )
    
    parser.add_argument(
        '--no-load',
        action='store_true',
        help='Skip loading the knowledge base on startup'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize agent
    console.print("[bold]Initializing agent...[/bold]")
    agent = initialize_agent(args)
    console.print(f"[green]✓ Agent '{agent.name}' ready[/green]")
    
    # Load knowledge base
    if not args.no_load:
        load_knowledge_base(agent, args)
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(agent, args.query)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()

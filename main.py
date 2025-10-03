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

# Global verbosity level
VERBOSITY = "normal"  # "quiet", "normal", "verbose"


def set_verbosity(level):
    """Set the global verbosity level."""
    global VERBOSITY
    VERBOSITY = level.lower()


def print_status(message, level="normal", style=""):
    """Print status message based on verbosity level."""
    if VERBOSITY == "quiet":
        return
    if level == "verbose" and VERBOSITY != "verbose":
        return
    
    if style:
        console.print(f"[{style}]{message}[/{style}]")
    else:
        console.print(message)


def print_error(message):
    """Always print errors regardless of verbosity."""
    console.print(f"[red]{message}[/red]")


def print_success(message, level="normal"):
    """Print success message based on verbosity level."""
    print_status(message, level, "green")


def print_warning(message, level="normal"):
    """Print warning message based on verbosity level."""
    print_status(message, level, "yellow")


def print_banner():
    """Print welcome banner."""
    if VERBOSITY == "quiet":
        return
        
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
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        agent = AIAgent(config_path)
        
        # Set verbosity from agent config if not overridden by command line
        if not hasattr(args, 'verbosity') or args.verbosity is None:
            config_verbosity = getattr(agent, 'verbosity', 'normal')
            set_verbosity(config_verbosity)
        
        print_status(f"Agent '{agent.name}' initialized", "verbose")
        return agent
    except Exception as e:
        print_error(f"Error initializing agent: {e}")
        sys.exit(1)


def load_knowledge_base(agent, args):
    """Load the knowledge base."""
    kb_path = args.kb_path or "./rag_db"
    
    if not os.path.exists(kb_path):
        print_warning(f"Knowledge base directory not found: {kb_path}")
        print_status("Creating directory...", "verbose")
        os.makedirs(kb_path, exist_ok=True)
        print_warning("Please add markdown files to this directory.")
        return False
    
    try:
        if VERBOSITY != "quiet":
            with console.status("[bold green]Loading knowledge base..."):
                agent.load_knowledge_base(kb_path)
        else:
            agent.load_knowledge_base(kb_path)
        
        print_success("Knowledge base loaded", "normal")
        return True
    except Exception as e:
        print_error(f"Error loading knowledge base: {e}")
        return False


def interactive_mode(agent):
    """Run the agent in interactive mode."""
    if VERBOSITY != "quiet":
        console.print("\n[bold green]Interactive Mode[/bold green]")
        console.print("Type your questions. Commands: /help, /clear, /quit\n")
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if not query.strip():
                continue
            
            # Handle commands
            if query.lower() in ['/quit', '/exit']:
                if VERBOSITY != "quiet":
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
                print_status("Knowledge base cleared", "normal", "yellow")
                continue
            
            # Process query
            if VERBOSITY != "quiet":
                with console.status("[bold green]Thinking..."):
                    result = agent.chat(query)
            else:
                result = agent.chat(query)
            
            # Display response
            console.print(f"\n[bold magenta]{agent.name}[/bold magenta]:")
            console.print(Markdown(result['response']))
            
            # Display sources if any and if verbose enough
            if result['sources'] and VERBOSITY != "quiet":
                sources_str = ", ".join(set(s['source'] for s in result['sources']))
                console.print(f"\n[dim]Sources: {sources_str}[/dim]")
            
            console.print()
            
        except KeyboardInterrupt:
            if VERBOSITY != "quiet":
                console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            print_error(f"Error: {e}")


def single_query_mode(agent, query):
    """Process a single query and exit."""
    try:
        if VERBOSITY != "quiet":
            with console.status("[bold green]Processing query..."):
                result = agent.chat(query)
        else:
            result = agent.chat(query)
        
        # In quiet mode, just show the response without formatting
        if VERBOSITY == "quiet":
            console.print(result['response'])
        else:
            console.print("\n[bold magenta]Response:[/bold magenta]")
            console.print(Markdown(result['response']))
            
            if result['sources']:
                sources_str = ", ".join(set(s['source'] for s in result['sources']))
                console.print(f"\n[dim]Sources: {sources_str}[/dim]")
            
            console.print()
        
    except Exception as e:
        print_error(f"Error: {e}")
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
  %(prog)s --quiet -q "question"     # Minimal output
  %(prog)s --verbose                 # Detailed output
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
    
    # Verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--quiet', '-q-mode',
        action='store_const',
        const='quiet',
        dest='verbosity',
        help='Minimal output (quiet mode)'
    )
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_const',
        const='verbose',
        dest='verbosity',
        help='Detailed output (verbose mode)'
    )
    
    args = parser.parse_args()
    
    # Set verbosity from command line if provided
    if args.verbosity:
        set_verbosity(args.verbosity)
    
    # Print banner
    print_banner()
    
    # Initialize agent
    print_status("Initializing agent...", "normal", "bold")
    agent = initialize_agent(args)
    print_success(f"Agent '{agent.name}' ready", "normal")
    
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

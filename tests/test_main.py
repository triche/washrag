"""
Tests for the main CLI application.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import main


class TestMainCLI:
    """Test suite for main CLI application."""
    
    def test_print_banner(self, capsys):
        """Test that banner prints correctly."""
        main.print_banner()
        captured = capsys.readouterr()
        
        assert "WashRAG" in captured.out
        assert "Basic RAG System" in captured.out
    
    @patch('main.AIAgent')
    def test_initialize_agent_success(self, mock_agent_class, config_file):
        """Test successful agent initialization."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Create args mock
        args = Mock()
        args.config = config_file
        
        result = main.initialize_agent(args)
        
        assert result == mock_agent
        mock_agent_class.assert_called_once_with(config_file)
    
    def test_initialize_agent_missing_config(self):
        """Test agent initialization with missing config file."""
        args = Mock()
        args.config = "/path/that/does/not/exist.yaml"
        
        with pytest.raises(SystemExit) as exc_info:
            main.initialize_agent(args)
        
        assert exc_info.value.code == 1
    
    @patch('main.AIAgent')
    def test_initialize_agent_with_exception(self, mock_agent_class):
        """Test agent initialization that raises exception."""
        mock_agent_class.side_effect = Exception("Initialization failed")
        
        args = Mock()
        args.config = None  # Will use default
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                main.initialize_agent(args)
            
            assert exc_info.value.code == 1
    
    def test_load_knowledge_base_success(self, test_markdown_files):
        """Test successful knowledge base loading."""
        mock_agent = Mock()
        
        args = Mock()
        args.kb_path = test_markdown_files
        
        result = main.load_knowledge_base(mock_agent, args)
        
        assert result is True
        mock_agent.load_knowledge_base.assert_called_once_with(test_markdown_files)
    
    def test_load_knowledge_base_missing_directory(self, temp_dir):
        """Test knowledge base loading with missing directory."""
        mock_agent = Mock()
        
        args = Mock()
        args.kb_path = os.path.join(temp_dir, "nonexistent")
        
        with patch('os.makedirs') as mock_makedirs:
            result = main.load_knowledge_base(mock_agent, args)
        
        assert result is False
        mock_makedirs.assert_called_once()
    
    def test_load_knowledge_base_with_exception(self, test_markdown_files):
        """Test knowledge base loading that raises exception."""
        mock_agent = Mock()
        mock_agent.load_knowledge_base.side_effect = Exception("Loading failed")
        
        args = Mock()
        args.kb_path = test_markdown_files
        
        result = main.load_knowledge_base(mock_agent, args)
        
        assert result is False
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_quit_command(self, mock_ask):
        """Test interactive mode quit command."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        
        # Simulate user typing /quit
        mock_ask.return_value = "/quit"
        
        # Should exit without error
        main.interactive_mode(mock_agent)
        
        # Agent should not be called for /quit command
        mock_agent.chat.assert_not_called()
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_help_command(self, mock_ask, capsys):
        """Test interactive mode help command."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        
        # Simulate user typing /help then /quit
        mock_ask.side_effect = ["/help", "/quit"]
        
        main.interactive_mode(mock_agent)
        
        captured = capsys.readouterr()
        assert "Available Commands" in captured.out
        assert "/help" in captured.out
        assert "/clear" in captured.out
        assert "/quit" in captured.out
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_clear_command(self, mock_ask):
        """Test interactive mode clear command."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        
        # Simulate user typing /clear then /quit
        mock_ask.side_effect = ["/clear", "/quit"]
        
        main.interactive_mode(mock_agent)
        
        mock_agent.clear_knowledge_base.assert_called_once()
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_normal_query(self, mock_ask):
        """Test interactive mode with normal query."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_agent.chat.return_value = {
            'response': 'Test response',
            'sources': [{'source': 'test.md'}]
        }
        
        # Simulate user query then quit
        mock_ask.side_effect = ["What is Python?", "/quit"]
        
        main.interactive_mode(mock_agent)
        
        mock_agent.chat.assert_called_once_with("What is Python?")
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_empty_query(self, mock_ask):
        """Test interactive mode with empty query."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        
        # Simulate empty input then quit
        mock_ask.side_effect = ["", "   ", "/quit"]
        
        main.interactive_mode(mock_agent)
        
        # Should not call agent for empty queries
        mock_agent.chat.assert_not_called()
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_keyboard_interrupt(self, mock_ask):
        """Test interactive mode with keyboard interrupt."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        
        # Simulate KeyboardInterrupt (Ctrl+C)
        mock_ask.side_effect = KeyboardInterrupt()
        
        # Should exit gracefully
        main.interactive_mode(mock_agent)
    
    @patch('main.Prompt.ask')
    def test_interactive_mode_exception_handling(self, mock_ask):
        """Test interactive mode exception handling."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_agent.chat.side_effect = Exception("Chat error")
        
        # Simulate query that causes exception, then quit
        mock_ask.side_effect = ["test query", "/quit"]
        
        # Should handle exception gracefully
        main.interactive_mode(mock_agent)
    
    def test_single_query_mode_success(self):
        """Test single query mode with successful response."""
        mock_agent = Mock()
        mock_agent.chat.return_value = {
            'response': 'Test response',
            'sources': [{'source': 'test.md'}]
        }
        
        main.single_query_mode(mock_agent, "What is Python?")
        
        mock_agent.chat.assert_called_once_with("What is Python?")
    
    def test_single_query_mode_no_sources(self):
        """Test single query mode with no sources."""
        mock_agent = Mock()
        mock_agent.chat.return_value = {
            'response': 'Test response',
            'sources': []
        }
        
        main.single_query_mode(mock_agent, "test query")
        
        mock_agent.chat.assert_called_once_with("test query")
    
    def test_single_query_mode_exception(self):
        """Test single query mode with exception."""
        mock_agent = Mock()
        mock_agent.chat.side_effect = Exception("Query failed")
        
        with pytest.raises(SystemExit) as exc_info:
            main.single_query_mode(mock_agent, "test query")
        
        assert exc_info.value.code == 1
    
    @patch('main.initialize_agent')
    @patch('main.load_knowledge_base')
    @patch('main.single_query_mode')
    @patch('main.print_banner')
    def test_main_single_query_mode(self, mock_banner, mock_single, mock_load, mock_init):
        """Test main function in single query mode."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_init.return_value = mock_agent
        mock_load.return_value = True
        
        test_args = [
            'main.py',
            '-q', 'What is Python?',
            '-c', 'config.yaml',
            '--kb-path', './docs'
        ]
        
        with patch('sys.argv', test_args):
            main.main()
        
        mock_banner.assert_called_once()
        mock_init.assert_called_once()
        mock_load.assert_called_once()
        mock_single.assert_called_once()
    
    @patch('main.initialize_agent')
    @patch('main.load_knowledge_base')
    @patch('main.interactive_mode')
    @patch('main.print_banner')
    def test_main_interactive_mode(self, mock_banner, mock_interactive, mock_load, mock_init):
        """Test main function in interactive mode."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_init.return_value = mock_agent
        mock_load.return_value = True
        
        test_args = ['main.py']  # No query argument
        
        with patch('sys.argv', test_args):
            main.main()
        
        mock_banner.assert_called_once()
        mock_init.assert_called_once()
        mock_load.assert_called_once()
        mock_interactive.assert_called_once()
    
    @patch('main.initialize_agent')
    @patch('main.load_knowledge_base')
    @patch('main.interactive_mode')
    @patch('main.print_banner')
    def test_main_no_load_flag(self, mock_banner, mock_interactive, mock_load, mock_init):
        """Test main function with --no-load flag."""
        mock_agent = Mock()
        mock_agent.name = "Test Agent"
        mock_init.return_value = mock_agent
        
        test_args = ['main.py', '--no-load']
        
        with patch('sys.argv', test_args):
            main.main()
        
        mock_banner.assert_called_once()
        mock_init.assert_called_once()
        mock_load.assert_not_called()  # Should not load KB
        mock_interactive.assert_called_once()
    
    def test_argument_parser(self):
        """Test argument parser configuration."""
        parser = main.argparse.ArgumentParser()
        
        # Add the same arguments as in main()
        parser.add_argument('-q', '--query', help='Single query to process')
        parser.add_argument('-c', '--config', help='Path to config file')
        parser.add_argument('--kb-path', help='Path to knowledge base directory')
        parser.add_argument('--no-load', action='store_true', help='Skip loading KB')
        
        # Test parsing different argument combinations
        args1 = parser.parse_args(['-q', 'test query'])
        assert args1.query == 'test query'
        assert args1.config is None
        assert args1.no_load is False
        
        args2 = parser.parse_args(['--config', 'test.yaml', '--no-load'])
        assert args2.config == 'test.yaml'
        assert args2.no_load is True
        
        args3 = parser.parse_args(['--kb-path', './docs'])
        assert args3.kb_path == './docs'
    
    def test_default_config_path(self):
        """Test default configuration path handling."""
        args = Mock()
        args.config = None  # No config specified
        
        with patch('os.path.exists', return_value=True):
            with patch('main.AIAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                
                result = main.initialize_agent(args)
                
                # Should use default config path
                mock_agent_class.assert_called_once_with("./config/agent_config.yaml")
    
    def test_default_kb_path(self, temp_dir):
        """Test default knowledge base path handling."""
        mock_agent = Mock()
        
        args = Mock()
        args.kb_path = None  # No KB path specified
        
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs') as mock_makedirs:
                result = main.load_knowledge_base(mock_agent, args)
        
        # Should use default KB path
        mock_makedirs.assert_called_once_with("./rag_db", exist_ok=True)

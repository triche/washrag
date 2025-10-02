#!/usr/bin/env python3
"""
Simple test demo script to show the pytest test suite functionality.
"""

import os
import sys
import subprocess
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_test_suite():
    """Demonstrate the test suite capabilities."""
    print("🧪 WashRAG Test Suite Demo")
    print("=" * 50)
    
    # Check if we can import the modules
    try:
        from rag_database import RAGDatabase
        from agent import AIAgent
        print("✅ Successfully imported WashRAG modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False
    
    # Try to run a simple test
    print("\n🔧 Running sample tests...")
    
    try:
        # Test RAG Database initialization
        temp_db = tempfile.mkdtemp()
        rag_db = RAGDatabase(
            db_path=temp_db,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=20
        )
        print("✅ RAGDatabase initialization test passed")
        
        # Test text chunking
        test_text = "This is a test document. " * 50
        chunks = rag_db.chunk_text(test_text, "test.md")
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        print(f"✅ Text chunking test passed ({len(chunks)} chunks created)")
        
        # Test configuration loading
        import yaml
        config_path = "./config/agent_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert 'agent' in config
            assert 'rag' in config
            assert 'llm' in config
            print("✅ Configuration loading test passed")
        else:
            print("⚠️  Configuration file not found, skipping config test")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_db, ignore_errors=True)
        
        print("\n🎉 All demo tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_test_structure():
    """Show the test suite structure."""
    print("\n📁 Test Suite Structure:")
    print("=" * 30)
    
    test_files = [
        ("tests/test_rag_database.py", "RAG Database component tests"),
        ("tests/test_agent.py", "AI Agent component tests"),
        ("tests/test_main.py", "CLI application tests"),
        ("tests/test_integration.py", "Integration tests"),
        ("tests/test_performance.py", "Performance tests"),
        ("tests/test_utils.py", "Test utilities and helpers"),
        ("tests/conftest.py", "Pytest configuration and fixtures"),
    ]
    
    for filename, description in test_files:
        if os.path.exists(filename):
            print(f"✅ {filename:<30} - {description}")
        else:
            print(f"❌ {filename:<30} - {description}")

def show_available_commands():
    """Show available test commands."""
    print("\n🚀 Available Test Commands:")
    print("=" * 30)
    
    commands = [
        ("python run_tests.py", "Run quick tests (unit + integration)"),
        ("python run_tests.py unit", "Run unit tests only"),
        ("python run_tests.py integration", "Run integration tests"),
        ("python run_tests.py all", "Run all tests including performance"),
        ("python run_tests.py coverage", "Run tests with coverage report"),
        ("pytest tests/", "Run all tests directly with pytest"),
        ("pytest -m 'not slow'", "Run tests excluding slow performance tests"),
        ("pytest --cov=src", "Run tests with coverage"),
    ]
    
    for command, description in commands:
        print(f"  {command:<35} - {description}")

def main():
    """Main demo function."""
    print("WashRAG Test Suite")
    print("==================\n")
    
    # Show test structure
    show_test_structure()
    
    # Show available commands
    show_available_commands()
    
    # Run demo tests
    success = demo_test_suite()
    
    if success:
        print("\n✨ Test suite is ready to use!")
        print("\nTo run the full test suite, use:")
        print("  python run_tests.py")
        print("\nFor more options, see:")
        print("  python run_tests.py --help")
    else:
        print("\n⚠️  Some issues detected. Please check the setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

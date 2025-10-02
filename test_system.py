#!/usr/bin/env python3
"""
Simple test script to verify the WashRAG system components.
Tests RAG database functionality without requiring an API key.
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_database import RAGDatabase


def test_rag_database():
    """Test RAG database functionality."""
    print("Testing RAG Database...")
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    temp_db = tempfile.mkdtemp()
    
    try:
        # Create test markdown file
        test_content = """# Test Document

This is a test document for the RAG system.

## Section 1

This section contains information about Python programming.
Python is a high-level programming language.

## Section 2

This section contains information about RAG systems.
RAG stands for Retrieval Augmented Generation.
"""
        
        test_file = os.path.join(temp_dir, "test.md")
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"✓ Created test file: {test_file}")
        
        # Initialize RAG database
        rag_db = RAGDatabase(
            db_path=temp_db,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20
        )
        print("✓ Initialized RAG database")
        
        # Load markdown files
        rag_db.load_markdown_files(temp_dir)
        print("✓ Loaded markdown files")
        
        # Get stats
        stats = rag_db.get_stats()
        print(f"✓ Database stats: {stats}")
        
        # Test query
        results = rag_db.query("What is Python?", top_k=2)
        print(f"✓ Query returned {len(results)} results")
        
        if results:
            print("\nTop result:")
            text, similarity, metadata = results[0]
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Source: {metadata['source']}")
            print(f"  Text: {text[:100]}...")
        
        # Test another query
        results = rag_db.query("What is RAG?", top_k=2)
        print(f"✓ Second query returned {len(results)} results")
        
        if results:
            print("\nTop result:")
            text, similarity, metadata = results[0]
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Source: {metadata['source']}")
            print(f"  Text: {text[:100]}...")
        
        print("\n✅ All RAG database tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(temp_db, ignore_errors=True)


def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting Configuration Loading...")
    
    import yaml
    
    config_path = "./config/agent_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['agent', 'rag', 'llm']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section in config: {section}")
                return False
            print(f"✓ Found config section: {section}")
        
        # Check agent settings
        agent_config = config['agent']
        print(f"  Agent name: {agent_config['name']}")
        print(f"  Temperature: {agent_config['temperature']}")
        
        # Check RAG settings
        rag_config = config['rag']
        print(f"  Embedding model: {rag_config['embedding_model']}")
        print(f"  Top K: {rag_config['top_k']}")
        
        print("✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_knowledge_base_files():
    """Test that knowledge base files exist and are readable."""
    print("\nTesting Knowledge Base Files...")
    
    kb_path = "./rag_db"
    
    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base directory not found: {kb_path}")
        return False
    
    md_files = [f for f in os.listdir(kb_path) if f.endswith('.md')]
    
    if not md_files:
        print(f"❌ No markdown files found in {kb_path}")
        return False
    
    print(f"✓ Found {len(md_files)} markdown files:")
    
    for filename in md_files:
        filepath = os.path.join(kb_path, filename)
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            print(f"  ✓ {filename} ({len(content)} characters)")
        except Exception as e:
            print(f"  ❌ Error reading {filename}: {e}")
            return False
    
    print("✅ Knowledge base files test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WashRAG System Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Knowledge Base Files", test_knowledge_base_files),
        ("RAG Database", test_rag_database),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

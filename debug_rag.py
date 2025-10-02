#!/usr/bin/env python3
"""
Debug script to test RAGDatabase functionality directly.
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_rag_database_import():
    """Test importing RAGDatabase."""
    try:
        from rag_database import RAGDatabase
        print("✅ Successfully imported RAGDatabase")
        return True
    except ImportError as e:
        print(f"❌ Failed to import RAGDatabase: {e}")
        return False

def test_rag_database_initialization():
    """Test RAGDatabase initialization."""
    try:
        from rag_database import RAGDatabase
        
        temp_db = tempfile.mkdtemp()
        print(f"Created temp db directory: {temp_db}")
        
        rag_db = RAGDatabase(
            db_path=temp_db,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20
        )
        
        print("✅ Successfully initialized RAGDatabase")
        print(f"  - DB path: {rag_db.db_path}")
        print(f"  - Chunk size: {rag_db.chunk_size}")
        print(f"  - Chunk overlap: {rag_db.chunk_overlap}")
        
        # Cleanup
        shutil.rmtree(temp_db, ignore_errors=True)
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAGDatabase: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_chunking():
    """Test text chunking."""
    try:
        from rag_database import RAGDatabase
        
        temp_db = tempfile.mkdtemp()
        
        rag_db = RAGDatabase(
            db_path=temp_db,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20
        )
        
        test_text = "This is a test sentence. " * 20
        chunks = rag_db.chunk_text(test_text, "test.md")
        
        print(f"✅ Successfully chunked text into {len(chunks)} chunks")
        
        # Cleanup
        shutil.rmtree(temp_db, ignore_errors=True)
        return True
        
    except Exception as e:
        print(f"❌ Failed text chunking: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests."""
    print("🔍 RAGDatabase Debug Tests")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_rag_database_import),
        ("Initialization Test", test_rag_database_initialization),
        ("Chunking Test", test_simple_chunking),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All debug tests passed! The issue might be with pytest setup.")
    else:
        print("💥 Some debug tests failed. This indicates a module issue.")

if __name__ == "__main__":
    main()

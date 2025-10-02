#!/usr/bin/env python3
"""
Simple test to check what's going wrong.
"""

import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Add src to path
src_path = os.path.join(os.getcwd(), 'src')
print("Adding to path:", src_path)
sys.path.insert(0, src_path)

print("Python path (first 5):", sys.path[:5])

try:
    print("Attempting to import rag_database...")
    import rag_database
    print("✅ Successfully imported rag_database module")
    
    print("Attempting to import RAGDatabase class...")
    from rag_database import RAGDatabase
    print("✅ Successfully imported RAGDatabase class")
    
    print("Attempting to create instance...")
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    
    # Try to create instance
    db = RAGDatabase(db_path=temp_dir)
    print("✅ Successfully created RAGDatabase instance")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

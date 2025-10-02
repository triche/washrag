"""
Minimal test to check RAGDatabase functionality.
"""

import os
import sys
import tempfile
import shutil

# Add path properly
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def test_basic_import():
    """Test basic import and initialization."""
    print("Testing basic import and initialization...")
    
    try:
        print("1. Importing RAGDatabase...")
        from rag_database import RAGDatabase
        print("   ✅ Import successful")
        
        print("2. Creating temporary directory...")
        temp_dir = tempfile.mkdtemp()
        print(f"   ✅ Created: {temp_dir}")
        
        print("3. Initializing RAGDatabase...")
        # Use a simpler initialization
        db = RAGDatabase(db_path=temp_dir)
        print("   ✅ Initialization successful")
        
        print("4. Testing chunk_text method...")
        test_text = "This is a test sentence."
        chunks = db.chunk_text(test_text, "test.md")
        print(f"   ✅ Created {len(chunks)} chunks")
        
        print("5. Cleaning up...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("   ✅ Cleanup successful")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Other error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Minimal RAGDatabase Test")
    print("=" * 30)
    success = test_basic_import()
    if success:
        print("\n🎉 Basic functionality works!")
        print("The issue might be with the pytest configuration.")
    else:
        print("\n💥 Basic functionality failed!")
        print("Need to fix the core modules first.")

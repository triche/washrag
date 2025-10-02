#!/usr/bin/env python3
"""
Quick validation script for WashRAG system structure.
Verifies the system is properly set up without requiring network access.
"""

import os
import sys
import yaml


def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False


def check_directory(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False


def validate_config(config_path):
    """Validate the configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = {
            'agent': ['name', 'personality', 'system_prompt', 'temperature', 'max_tokens'],
            'rag': ['chunk_size', 'chunk_overlap', 'top_k', 'similarity_threshold', 'embedding_model'],
            'llm': ['provider', 'model', 'api_key_env']
        }
        
        for section, keys in required_sections.items():
            if section not in config:
                print(f"  ✗ Missing section: {section}")
                return False
            
            for key in keys:
                if key not in config[section]:
                    print(f"  ✗ Missing key in {section}: {key}")
                    return False
        
        print(f"  ✓ All required configuration sections present")
        print(f"  ✓ Agent: {config['agent']['name']}")
        print(f"  ✓ Model: {config['llm']['model']}")
        print(f"  ✓ Embedding: {config['rag']['embedding_model']}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error validating config: {e}")
        return False


def check_python_syntax(files):
    """Check Python syntax."""
    import py_compile
    
    all_ok = True
    for filepath in files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"  ✓ {os.path.basename(filepath)}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {os.path.basename(filepath)}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main validation."""
    print("=" * 60)
    print("WashRAG System Validation")
    print("=" * 60)
    
    checks = []
    
    # Check project structure
    print("\n1. Project Structure")
    checks.append(check_directory("./config", "Config directory"))
    checks.append(check_directory("./rag_db", "RAG database directory"))
    checks.append(check_directory("./src", "Source directory"))
    
    # Check essential files
    print("\n2. Essential Files")
    checks.append(check_file("./config/agent_config.yaml", "Agent configuration"))
    checks.append(check_file("./requirements.txt", "Requirements file"))
    checks.append(check_file("./.env.example", "Environment template"))
    checks.append(check_file("./main.py", "Main CLI script"))
    checks.append(check_file("./src/agent.py", "Agent module"))
    checks.append(check_file("./src/rag_database.py", "RAG database module"))
    
    # Check knowledge base
    print("\n3. Knowledge Base")
    if os.path.exists("./rag_db"):
        md_files = [f for f in os.listdir("./rag_db") if f.endswith('.md')]
        if md_files:
            print(f"  ✓ Found {len(md_files)} markdown files:")
            for f in md_files:
                print(f"    - {f}")
            checks.append(True)
        else:
            print("  ✗ No markdown files in rag_db/")
            checks.append(False)
    else:
        checks.append(False)
    
    # Validate configuration
    print("\n4. Configuration File")
    if os.path.exists("./config/agent_config.yaml"):
        checks.append(validate_config("./config/agent_config.yaml"))
    else:
        checks.append(False)
    
    # Check Python syntax
    print("\n5. Python Syntax")
    python_files = [
        "./main.py",
        "./src/agent.py",
        "./src/rag_database.py",
        "./src/__init__.py"
    ]
    checks.append(check_python_syntax(python_files))
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for c in checks if c)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All {total} validation checks passed!")
        print("\n📚 To use WashRAG:")
        print("  1. Copy .env.example to .env and add your OpenAI API key")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Run the agent: python main.py")
        print("\n📝 Add more markdown files to rag_db/ to expand the knowledge base")
        return 0
    else:
        print(f"⚠️  {total - passed} validation check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test runner script for WashRAG application.

This script provides convenient commands to run different types of tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="WashRAG Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test categories:
  unit         - Run unit tests only
  integration  - Run integration tests only
  performance  - Run performance tests (slow)
  all          - Run all tests
  quick        - Run quick tests (unit + integration, no performance)
  coverage     - Run tests with coverage report
  
Examples:
  %(prog)s unit              # Run unit tests
  %(prog)s integration       # Run integration tests  
  %(prog)s all               # Run all tests
  %(prog)s coverage          # Run with coverage
  %(prog)s --verbose unit    # Run unit tests with verbose output
        """
    )
    
    parser.add_argument(
        'test_type',
        choices=['unit', 'integration', 'performance', 'all', 'quick', 'coverage'],
        nargs='?',
        default='quick',
        help='Type of tests to run (default: quick)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '-k', '--keyword',
        help='Run tests matching keyword expression'
    )
    
    parser.add_argument(
        '--no-cov',
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML coverage report'
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Base pytest command
    python_exe = sys.executable
    pytest_cmd = [python_exe, '-m', 'pytest']
    
    if args.verbose:
        pytest_cmd.append('-v')
    else:
        pytest_cmd.append('-q')
    
    if args.keyword:
        pytest_cmd.extend(['-k', args.keyword])
    
    # Configure based on test type
    description = "Tests"  # Default description
    
    if args.test_type == 'unit':
        pytest_cmd.extend(['tests/', '-m', 'not integration and not slow'])
        description = "Unit Tests"
        
    elif args.test_type == 'integration':
        pytest_cmd.extend(['tests/test_integration.py'])
        description = "Integration Tests"
        
    elif args.test_type == 'performance':
        pytest_cmd.extend(['tests/test_performance.py', '-m', 'slow'])
        description = "Performance Tests"
        
    elif args.test_type == 'all':
        pytest_cmd.append('tests/')
        description = "All Tests"
        
    elif args.test_type == 'quick':
        pytest_cmd.extend(['tests/', '-m', 'not slow'])
        description = "Quick Tests (Unit + Integration)"
        
    elif args.test_type == 'coverage':
        pytest_cmd.extend([
            'tests/',
            '--cov=src',
            '--cov-report=term-missing'
        ])
        if args.html:
            pytest_cmd.append('--cov-report=html:htmlcov')
        description = "Coverage Tests"
    
    # Add coverage if not disabled and not already added
    if not args.no_cov and args.test_type != 'coverage' and args.test_type != 'performance':
        pytest_cmd.extend(['--cov=src', '--cov-report=term'])
    
    # Run the tests
    success = run_command(pytest_cmd, description)
    
    if success:
        print(f"\n🎉 All {description.lower()} passed!")
        
        if args.html and (args.test_type == 'coverage' or not args.no_cov):
            print(f"\n📊 HTML coverage report generated: {project_dir}/htmlcov/index.html")
    else:
        print(f"\n💥 Some {description.lower()} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run All Tests for py_ntcpx v1.0
================================

Master test runner that executes all test suites and generates comprehensive reports.

Usage:
    python run_all_tests.py [--output_dir test_reports] [--verbose]

Software: py_ntcpx v1.0
"""

import sys
import argparse
from pathlib import Path

# Windows-safe encoding configuration
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

# Import Windows-safe utilities
try:
    from windows_safe_utils import safe_encode_unicode, safe_print
except ImportError:
    def safe_encode_unicode(text):
        replacements = {
            '✓': '[OK]', '✗': '[FAIL]', '✔': '[PASS]', '❌': '[ERROR]',
            '→': '->', '↳': '->', '±': '+/-', 'μ': 'mu', 'σ': 'sigma'
        }
        result = str(text)
        for u, a in replacements.items():
            result = result.replace(u, a)
        return result
    def safe_print(*args, **kwargs):
        safe_args = [safe_encode_unicode(str(arg)) for arg in args]
        print(*safe_args, **kwargs)

# Import test runners
try:
    from test_script_runner import TestReporter
except ImportError:
    print("Error: test_script_runner.py not found")
    sys.exit(1)

try:
    import test_ntcp_pipeline
except ImportError:
    test_ntcp_pipeline = None
    print("Warning: test_ntcp_pipeline.py not found")

try:
    import test_data_validation
except ImportError:
    test_data_validation = None
    print("Warning: test_data_validation.py not found")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Run all tests for py_ntcpx v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs comprehensive test suites:
- Unit tests for all modules
- Integration tests
- Data validation tests
- Script execution tests
- Module import tests
- File structure tests

Generates detailed test reports in JSON and text formats.

Software: py_ntcpx v1.0
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_reports',
        help='Output directory for test reports (default: test_reports)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create test reporter
    reporter = TestReporter(args.output_dir)
    
    # Run all tests
    success = reporter.run_all_tests()
    
    # Print final status
    if success:
        safe_print("\n[OK] All tests passed!")
        return 0
    else:
        safe_print("\n[FAIL] Some tests failed. Check test reports for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


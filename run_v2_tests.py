"""
Test runner for py_ntcpx v2.0 components

Ensures all tests use synthetic data (no real patient data)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Generate synthetic data first
print("Generating synthetic test data...")
try:
    from tests.test_data.generate_synthetic_data import generate_full_test_dataset
    test_data_dir = Path('tests/test_data/synthetic')
    generate_full_test_dataset(test_data_dir)
    print(f"Synthetic data generated at: {test_data_dir}\n")
except Exception as e:
    print(f"Warning: Could not generate synthetic data: {e}")
    print("Continuing with existing test data...\n")

# Run tests
import unittest

if __name__ == '__main__':
    print("=" * 60)
    print("Running py_ntcpx v2.0 Test Suite")
    print("=" * 60)
    print()
    
    # Load test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    try:
        from tests.test_data_splitter import TestPatientDataSplitter
        suite.addTests(loader.loadTestsFromTestCase(TestPatientDataSplitter))
        print("Added: TestPatientDataSplitter")
    except ImportError as e:
        print(f"Warning: Could not import TestPatientDataSplitter: {e}")
    
    print()
    print("=" * 60)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests completed with {len(result.failures)} failures and {len(result.errors)} errors")
    print("=" * 60)
    
    sys.exit(0 if result.wasSuccessful() else 1)

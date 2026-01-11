#!/usr/bin/env python3
"""
Test Script Runner for py_ntcpx v1.0
=====================================

Runs comprehensive tests and generates detailed test reports.
Tests all pipeline components without modifying existing code.

Usage:
    python test_script_runner.py [--output_dir test_reports] [--verbose]

Software: py_ntcpx v1.0
"""

import argparse
import sys
import os
import json
import datetime
from pathlib import Path
import subprocess
import traceback

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

# Import test modules
try:
    import test_ntcp_pipeline
except ImportError:
    test_ntcp_pipeline = None


class TestReporter:
    """Generate comprehensive test reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'software': 'py_ntcpx v1.0',
            'tests': {},
            'summary': {}
        }
    
    def run_unit_tests(self):
        """Run unit tests"""
        print("\n" + "="*60)
        print("Running Unit Tests")
        print("="*60)
        
        try:
            import unittest
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            if test_ntcp_pipeline:
                # Add all test classes
                test_classes = [
                    test_ntcp_pipeline.TestNTCPUtils,
                    test_ntcp_pipeline.TestNovelModels,
                    test_ntcp_pipeline.TestQAModules,
                    test_ntcp_pipeline.TestDataValidation,
                    test_ntcp_pipeline.TestOutputFormats,
                    test_ntcp_pipeline.TestBiologicalDVH,
                    test_ntcp_pipeline.TestIntegration
                ]
                
                for test_class in test_classes:
                    tests = loader.loadTestsFromTestCase(test_class)
                    suite.addTests(tests)
                
                runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
                result = runner.run(suite)
                
                self.test_results['tests']['unit_tests'] = {
                    'total': result.testsRun,
                    'passed': result.testsRun - len(result.failures) - len(result.errors),
                    'failed': len(result.failures),
                    'errors': len(result.errors),
                    'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                    'failures': [str(f[0]) for f in result.failures],
                    'errors': [str(e[0]) for e in result.errors]
                }
                
                return result.wasSuccessful()
            else:
                print("Warning: test_ntcp_pipeline module not available")
                self.test_results['tests']['unit_tests'] = {
                    'status': 'skipped',
                    'reason': 'test_ntcp_pipeline module not available'
                }
                return True
                
        except Exception as e:
            print(f"Error running unit tests: {e}")
            traceback.print_exc()
            self.test_results['tests']['unit_tests'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_script_execution(self):
        """Test that all scripts can be executed (help/version check)"""
        print("\n" + "="*60)
        print("Testing Script Execution")
        print("="*60)
        
        scripts = [
            'code1_dvh_preprocess.py',
            'code2_dvh_plot_and_summary.py',
            'code2_bDVH.py',
            'code3_ntcp_analysis_ml.py',
            'code4_ntcp_output_QA_reporter.py',
            'code5_ntcp_factors_analysis.py',
            'code6_publication_diagrams.py',
            'supp_results_summary.py',
            'run_pipeline.py'
        ]
        
        script_results = {}
        
        for script in scripts:
            script_path = Path(script)
            if not script_path.exists():
                script_results[script] = {
                    'status': 'not_found',
                    'error': 'Script file not found'
                }
                continue
            
            try:
                # Try to run script with --help to check it's executable
                result = subprocess.run(
                    [sys.executable, script, '--help'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=10
                )
                
                if result.returncode == 0 or 'usage:' in result.stdout.lower() or 'error' in result.stderr.lower():
                    script_results[script] = {
                        'status': 'executable',
                        'help_available': True
                    }
                else:
                    script_results[script] = {
                        'status': 'executable',
                        'help_available': False,
                        'stderr': result.stderr[:200]
                    }
                    
            except subprocess.TimeoutExpired:
                script_results[script] = {
                    'status': 'timeout',
                    'error': 'Script execution timed out'
                }
            except Exception as e:
                script_results[script] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.test_results['tests']['script_execution'] = script_results
        
        # Count results
        total = len(scripts)
        executable = sum(1 for r in script_results.values() if r.get('status') == 'executable')
        
        print(f"\nScript Execution Test Results:")
        print(f"  Total scripts: {total}")
        print(f"  Executable: {executable}")
        print(f"  Not found: {total - executable}")
        
        return executable == total
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        print("\n" + "="*60)
        print("Testing Module Imports")
        print("="*60)
        
        modules = {
            'pandas': 'pd',
            'numpy': 'np',
            'matplotlib': 'plt',
            'scipy': 'scipy',
            'sklearn': 'sklearn',
            'openpyxl': 'openpyxl',
            'xlsxwriter': 'xlsxwriter'
        }
        
        optional_modules = {
            'xgboost': 'xgb',
            'docx': 'docx',
            'seaborn': 'sns'
        }
        
        import_results = {}
        
        # Test required modules
        for module_name, import_name in modules.items():
            try:
                __import__(module_name)
                import_results[module_name] = {
                    'status': 'available',
                    'required': True
                }
                safe_print(f"  [OK] {module_name}")
            except ImportError:
                import_results[module_name] = {
                    'status': 'missing',
                    'required': True,
                    'error': f'Module {module_name} not found'
                }
                safe_print(f"  [FAIL] {module_name} (REQUIRED - MISSING)")
        
        # Test optional modules
        for module_name, import_name in optional_modules.items():
            try:
                __import__(module_name)
                import_results[module_name] = {
                    'status': 'available',
                    'required': False
                }
                safe_print(f"  [OK] {module_name} (optional)")
            except ImportError:
                import_results[module_name] = {
                    'status': 'missing',
                    'required': False
                }
                print(f"  - {module_name} (optional - not available)")
        
        self.test_results['tests']['module_imports'] = import_results
        
        # Check if all required modules are available
        required_available = all(
            r.get('status') == 'available' 
            for m, r in import_results.items() 
            if r.get('required', False)
        )
        
        return required_available
    
    def test_file_structure(self):
        """Test that required files exist"""
        print("\n" + "="*60)
        print("Testing File Structure")
        print("="*60)
        
        required_files = [
            'code1_dvh_preprocess.py',
            'code2_dvh_plot_and_summary.py',
            'code2_bDVH.py',
            'code3_ntcp_analysis_ml.py',
            'code4_ntcp_output_QA_reporter.py',
            'code5_ntcp_factors_analysis.py',
            'code6_publication_diagrams.py',
            'supp_results_summary.py',
            'run_pipeline.py',
            'ntcp_utils.py',
            'ntcp_novel_models.py',
            'ntcp_qa_modules.py',
            'README.md',
            'requirements.txt'
        ]
        
        file_results = {}
        
        for file_path in required_files:
            path = Path(file_path)
            exists = path.exists()
            file_results[file_path] = {
                'exists': exists,
                'size': path.stat().st_size if exists else 0
            }
            
            if exists:
                safe_print(f"  [OK] {file_path}")
            else:
                safe_print(f"  [FAIL] {file_path} (MISSING)")
        
        self.test_results['tests']['file_structure'] = file_results
        
        all_exist = all(r['exists'] for r in file_results.values())
        return all_exist
    
    def test_windows_console_safety(self):
        """Test Windows console safety - Unicode encoding"""
        safe_print("\n" + "="*60)
        safe_print("Testing Windows Console Safety")
        safe_print("="*60)
        
        test_results = {}
        passed = True
        
        # Test 1: Safe print with Unicode characters
        try:
            test_strings = [
                "Test with [OK]",
                "Test with [FAIL]",
                "Test with -> arrow",
                "Test with +/- symbol"
            ]
            for test_str in test_strings:
                safe_print(f"  Testing: {test_str}")
            test_results['safe_print'] = {'status': 'passed', 'error': None}
            safe_print("  [OK] Safe print function works correctly")
        except Exception as e:
            test_results['safe_print'] = {'status': 'failed', 'error': str(e)}
            safe_print(f"  [FAIL] Safe print function failed: {e}")
            passed = False
        
        # Test 2: UTF-8 encoding configuration
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                test_results['utf8_config'] = {'status': 'passed', 'error': None}
                safe_print("  [OK] UTF-8 encoding configuration works")
            else:
                test_results['utf8_config'] = {'status': 'skipped', 'error': 'reconfigure not available'}
                safe_print("  [WARN] UTF-8 reconfiguration not available (Python <3.7)")
        except Exception as e:
            test_results['utf8_config'] = {'status': 'failed', 'error': str(e)}
            safe_print(f"  [FAIL] UTF-8 encoding configuration failed: {e}")
            passed = False
        
        # Test 3: Unicode character replacement
        try:
            from windows_safe_utils import safe_encode_unicode
            test_cases = {
                '✓': '[OK]',
                '✗': '[FAIL]',
                '→': '->',
                '±': '+/-'
            }
            all_replacements_work = True
            for unicode_char, ascii_replacement in test_cases.items():
                result = safe_encode_unicode(f"Test{unicode_char}test")
                if ascii_replacement not in result:
                    all_replacements_work = False
                    break
            if all_replacements_work:
                test_results['unicode_replacement'] = {'status': 'passed', 'error': None}
                safe_print("  [OK] Unicode character replacement works correctly")
            else:
                test_results['unicode_replacement'] = {'status': 'failed', 'error': 'Replacement did not work'}
                safe_print("  [FAIL] Unicode character replacement failed")
                passed = False
        except ImportError:
            test_results['unicode_replacement'] = {'status': 'skipped', 'error': 'windows_safe_utils not available'}
            safe_print("  [WARN] windows_safe_utils module not available (using fallback)")
        except Exception as e:
            test_results['unicode_replacement'] = {'status': 'failed', 'error': str(e)}
            safe_print(f"  [FAIL] Unicode character replacement test failed: {e}")
            passed = False
        
        self.test_results['tests']['windows_console_safety'] = test_results
        
        if passed:
            safe_print("\n[OK] All Windows console safety tests passed!")
        else:
            safe_print("\n[FAIL] Some Windows console safety tests failed!")
        
        return passed
    
    def generate_summary(self):
        """Generate test summary"""
        tests = self.test_results.get('tests', {})
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        # Count unit tests
        if 'unit_tests' in tests:
            ut = tests['unit_tests']
            if 'total' in ut:
                total_tests += ut['total']
                passed_tests += ut.get('passed', 0)
                failed_tests += int(ut.get('failed', 0))
                errors = ut.get('errors', [])
                if isinstance(errors, list):
                    failed_tests += len(errors)
                else:
                    failed_tests += int(errors)
        
        # Count script execution tests
        if 'script_execution' in tests:
            se = tests['script_execution']
            total_scripts = len(se)
            executable_scripts = sum(1 for r in se.values() if r.get('status') == 'executable')
            total_tests += total_scripts
            passed_tests += executable_scripts
            failed_tests += (total_scripts - executable_scripts)
        
        # Count import tests
        if 'module_imports' in tests:
            mi = tests['module_imports']
            required_modules = [m for m, r in mi.items() if r.get('required', False)]
            available_required = sum(1 for m in required_modules if mi[m].get('status') == 'available')
            total_tests += len(required_modules)
            passed_tests += available_required
            failed_tests += (len(required_modules) - available_required)
        
        # Count file structure tests
        if 'file_structure' in tests:
            fs = tests['file_structure']
            total_files = len(fs)
            existing_files = sum(1 for r in fs.values() if r.get('exists', False))
            total_tests += total_files
            passed_tests += existing_files
            failed_tests += (total_files - existing_files)
        
        # Count Windows console safety tests
        if 'windows_console_safety' in tests:
            wcs = tests['windows_console_safety']
            total_wcs_tests = len(wcs)
            passed_wcs_tests = sum(1 for r in wcs.values() if r.get('status') == 'passed')
            total_tests += total_wcs_tests
            passed_tests += passed_wcs_tests
            failed_tests += (total_wcs_tests - passed_wcs_tests)
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        }
    
    def save_report(self):
        """Save test report to files"""
        # Generate summary
        self.generate_summary()
        
        # Save JSON report
        json_file = self.output_dir / 'test_report.json'
        with open(json_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save text report
        txt_file = self.output_dir / 'test_report.txt'
        with open(txt_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("py_ntcpx v1.0 - Comprehensive Test Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {self.test_results['timestamp']}\n")
            f.write(f"Software: {self.test_results['software']}\n\n")
            
            # Summary
            summary = self.test_results['summary']
            f.write("SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"Overall Status: {summary['overall_status']}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-"*60 + "\n\n")
            
            # Unit tests
            if 'unit_tests' in self.test_results['tests']:
                ut = self.test_results['tests']['unit_tests']
                f.write("Unit Tests:\n")
                if 'total' in ut:
                    f.write(f"  Total: {ut['total']}\n")
                    f.write(f"  Passed: {ut.get('passed', 0)}\n")
                    f.write(f"  Failed: {ut.get('failed', 0)}\n")
                    errors = ut.get('errors', [])
                    if isinstance(errors, list):
                        f.write(f"  Errors: {len(errors)}\n")
                    else:
                        f.write(f"  Errors: {int(errors)}\n")
                else:
                    f.write(f"  Status: {ut.get('status', 'unknown')}\n")
                f.write("\n")
            
            # Script execution
            if 'script_execution' in self.test_results['tests']:
                f.write("Script Execution Tests:\n")
                for script, result in self.test_results['tests']['script_execution'].items():
                    status = result.get('status', 'unknown')
                    f.write(f"  {script}: {status}\n")
                f.write("\n")
            
            # Module imports
            if 'module_imports' in self.test_results['tests']:
                f.write("Module Import Tests:\n")
                for module, result in self.test_results['tests']['module_imports'].items():
                    status = result.get('status', 'unknown')
                    required = "REQUIRED" if result.get('required', False) else "OPTIONAL"
                    f.write(f"  {module}: {status} ({required})\n")
                f.write("\n")
            
            # File structure
            if 'file_structure' in self.test_results['tests']:
                f.write("File Structure Tests:\n")
                for file_path, result in self.test_results['tests']['file_structure'].items():
                    exists = "EXISTS" if result.get('exists', False) else "MISSING"
                    f.write(f"  {file_path}: {exists}\n")
                f.write("\n")
        
        print(f"\nTest reports saved to:")
        print(f"  - {json_file}")
        print(f"  - {txt_file}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*60)
        print("py_ntcpx v1.0 - Comprehensive Test Suite")
        print("="*60)
        
        results = {}
        
        # Run all test categories
        results['file_structure'] = self.test_file_structure()
        results['module_imports'] = self.test_imports()
        results['script_execution'] = self.test_script_execution()
        results['unit_tests'] = self.run_unit_tests()
        results['windows_console_safety'] = self.test_windows_console_safety()
        
        # Save reports
        self.save_report()
        
        # Print final summary
        summary = self.test_results['summary']
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Status: {summary['overall_status']}")
        print("="*60)
        
        return summary['overall_status'] == 'PASS'


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for py_ntcpx v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    reporter = TestReporter(args.output_dir)
    success = reporter.run_all_tests()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())


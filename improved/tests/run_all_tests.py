#!/usr/bin/env python
"""
Meta test script to run all tests in the tests/ directory.

This script executes all individual test files and provides a comprehensive summary
of the test suite's performance.
"""

import unittest
import sys
import os
import subprocess
from datetime import datetime


def run_test_file(test_file):
    """Run a single test file and return the result."""
    print(f"\n{'='*70}")
    print(f"Running {test_file}...")
    print('='*70)
    
    # Run the test file as a subprocess
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout per test file
    )
    
    # Print the output (filter out internal unittest messages)
    output_lines = result.stdout.split('\n')
    filtered_output = []
    for line in output_lines:
        # Skip unittest internal messages that start with "test_"
        if line.strip().startswith('test_') and ('(' in line or '.' in line):
            continue
        filtered_output.append(line)
    
    print('\n'.join(filtered_output))
    
    # Only print stderr if it contains actual errors (not just unittest output)
    if result.stderr and not all(line.startswith('test_') for line in result.stderr.split('\n') if line.strip()):
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def run_all_tests():
    """Run all test files in the tests directory."""
    print("Starting Comprehensive Test Suite")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of test files to run
    test_files = [
        'test_simulator_randomness.py',
        'test_knitter_randomness.py',
        'test_knitter_symmetry.py',
        'test_simulator_noise.py',
        'test_knitter_noise.py'
    ]
    
    results = {}
    total_tests = len(test_files)
    passed_tests = 0
    
    for test_file in test_files:
        try:
            success = run_test_file(test_file)
            results[test_file] = success
            if success:
                passed_tests += 1
        except subprocess.TimeoutExpired:
            print(f"❌ {test_file} timed out after 5 minutes")
            results[test_file] = False
        except Exception as e:
            print(f"❌ {test_file} failed with exception: {e}")
            results[test_file] = False
    
    # Print summary
    print(f"\n{'='*70}")
    print("COMPREHENSIVE TEST SUMMARY")
    print('='*70)
    print(f"Tests completed: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nIndividual Test Results:")
    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_file}")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return overall success
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
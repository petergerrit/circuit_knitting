#!/usr/bin/env python3
"""
Run all quick Trotter step fermion number tests.

This script runs:
- test_initial_state_fermion_number.py (step 0)
- test_first_trotter_step_fermion_number.py (step 1)
- test_second_trotter_step_fermion_number.py (step 2)
"""

import subprocess
import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of test scripts to run
test_scripts = [
    "test_initial_state_fermion_number.py",
    "test_first_trotter_step_fermion_number.py",
    "test_second_trotter_step_fermion_number.py"
]

def run_test(script_name):
    """Run a single test script."""
    script_path = os.path.join(script_dir, script_name)
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=script_dir,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        return False
    return True

if __name__ == "__main__":
    print("Starting quick tests...")
    
    all_passed = True
    for script in test_scripts:
        if not run_test(script):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("All quick tests completed successfully!")
    else:
        print("Some tests failed!")
    print('='*60)
    
    sys.exit(0 if all_passed else 1)

#!/usr/bin/env python3
"""
Run all 8 knitted production scripts sequentially.
"""

import subprocess
import sys
import os

# List of all scripts to run
scripts = [
    "step1_knitted_eps0p2.py",
    "step1_knitted_eps0p4.py",
    "step1_knitted_eps0p6.py",
    "step1_knitted_eps0p8.py",
    "step2_knitted_eps0p5.py",
    "step2_knitted_eps0p6.py",
    "step2_knitted_eps0p7.py",
    "step2_knitted_eps0p8.py",
]

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the script directory
os.chdir(script_dir)

print(f"Running all knitted production scripts from: {script_dir}")
print("=" * 60)

for script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"\nRunning: {script}")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"✓ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ {script} failed with return code {e.returncode}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

print("\n" + "=" * 60)
print("All scripts completed successfully!")

#!/usr/bin/env python3
"""
Run all 9 knitted production scripts sequentially.
"""

import subprocess
import sys
import os

# List of all scripts to run
scripts = [
    "step0.py",
    "step1_eps0p2.py",
    "step1_eps0p4.py",
    "step1_eps0p6.py",
    "step1_eps0p8.py",
    "step2_eps0p5.py",
    "step2_eps0p6.py",
    "step2_eps0p7.py",
    "step2_eps0p8.py",
]

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the script directory
os.chdir(script_dir)

for script in scripts:
    script_path = os.path.join(script_dir, script)
    
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        sys.exit(1)

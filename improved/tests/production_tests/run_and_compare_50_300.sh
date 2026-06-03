#!/bin/bash
# Script to run test_circuits_50_300.py and compare results multiple times

NUM_RUNS=${1:-10}

echo "Running test_circuits_50_300.py $NUM_RUNS times and comparing each time..."
echo ""

for i in $(seq 1 $NUM_RUNS); do
    echo "=== Run $i/$NUM_RUNS ==="
    
    # Run the test script
    python3 test_circuits_50_300.py > /dev/null 2>&1
    
    # Compare results
    python3 compare_reproduction_50_300.py
    
    echo ""
done

echo "All runs completed."

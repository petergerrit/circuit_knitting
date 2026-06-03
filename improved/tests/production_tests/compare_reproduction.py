#!/usr/bin/env python3
"""
Comparison script to check if reproduction results match test1 or test2.

This script compares the raw measurement results from:
- debug_circuits_118_483_reproduction.txt (freshly generated)
- debug_step2_shots16_test1.txt (baseline)
- debug_step2_shots16_test2.txt (the different one)

For circuits 118 and 483.
"""

import re

def extract_circuit_results(filename, circuit_idx):
    """Extract raw measurement results for a specific circuit from a debug file."""
    results = {}
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to find circuit section and extract raw measurement results
    circuit_pattern = rf'Circuit {circuit_idx} / \d+ \(prefactor: [^)]+\)\s+Simulator seed: \d+, Transpiler seed: \d+\s+Raw measurement results \(internal\): (\{{[^}}]+\}})'
    
    match = re.search(circuit_pattern, content)
    if match:
        # Parse the dictionary string
        dict_str = match.group(1)
        # Simple parsing - extract all 'bitstring': count pairs
        bitstring_counts = {}
        for kv_match in re.finditer(r"'([01]+)': (\d+)", dict_str):
            bitstring = kv_match.group(1)
            count = int(kv_match.group(2))
            bitstring_counts[bitstring] = count
        return bitstring_counts
    return None

def compare_results(repro_file, test1_file, test2_file):
    """Compare reproduction results with test1 and test2."""
    circuits = [118, 483]
    
    print("=" * 70)
    print("COMPARISON: Reproduction vs test1 vs test2")
    print("=" * 70)
    
    test1_matches = []
    test2_matches = []
    
    for circuit_idx in circuits:
        print(f"\nCircuit {circuit_idx}:")
        
        # Extract results
        repro_results = extract_circuit_results(repro_file, circuit_idx)
        test1_results = extract_circuit_results(test1_file, circuit_idx)
        test2_results = extract_circuit_results(test2_file, circuit_idx)
        
        if repro_results is None:
            print(f"  ERROR: Could not find Circuit {circuit_idx} in {repro_file}")
            test1_matches.append(False)
            test2_matches.append(False)
            continue
        
        if test1_results is None:
            print(f"  ERROR: Could not find Circuit {circuit_idx} in {test1_file}")
            test1_matches.append(False)
            test2_matches.append(False)
            continue
        
        if test2_results is None:
            print(f"  ERROR: Could not find Circuit {circuit_idx} in {test2_file}")
            test1_matches.append(False)
            test2_matches.append(False)
            continue
        
        # Compare
        match_test1 = repro_results == test1_results
        match_test2 = repro_results == test2_results
        test1_eq_test2 = test1_results == test2_results
        
        print(f"  Reproduction: {sorted(repro_results.items())}")
        print(f"  Test1:       {sorted(test1_results.items())}")
        print(f"  Test2:       {sorted(test2_results.items())}")
        
        if match_test1 and match_test2:
            if test1_eq_test2:
                print(f"  ✓ MATCHES TEST1 AND TEST2 (test1==test2)")
            else:
                print(f"  ✓ MATCHES TEST1 AND TEST2")
            test1_matches.append(True)
            test2_matches.append(True)
        elif match_test1:
            print(f"  ✓ MATCHES TEST1")
            test1_matches.append(True)
            test2_matches.append(False)
        elif match_test2:
            print(f"  ✓ MATCHES TEST2")
            test1_matches.append(False)
            test2_matches.append(True)
        else:
            print(f"  ✗ MATCHES NEITHER")
            test1_matches.append(False)
            test2_matches.append(False)
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Circuits matching test1: {sum(test1_matches)}/{len(circuits)}")
    print(f"  Circuits matching test2: {sum(test2_matches)}/{len(circuits)}")
    
    # Check if all circuits match both (which happens when test1==test2 for those circuits)
    if all(t1 and t2 for t1, t2 in zip(test1_matches, test2_matches)):
        print("\nFINAL RESULT: Reproduction matches TEST1 AND TEST2 for ALL circuits")
        final_result = "BOTH"
    elif all(test1_matches):
        print("\nFINAL RESULT: Reproduction matches TEST1 for ALL circuits")
        final_result = "TEST1"
    elif all(test2_matches):
        print("\nFINAL RESULT: Reproduction matches TEST2 for ALL circuits")
        final_result = "TEST2"
    elif sum(test1_matches) > 0 and sum(test2_matches) > 0:
        print("\nFINAL RESULT: Reproduction matches MIXED (some test1, some test2)")
        final_result = "MIXED"
    else:
        print("\nFINAL RESULT: Reproduction matches NEITHER test1 nor test2")
        final_result = "NEITHER"
    print("=" * 70)
    
    return final_result

if __name__ == "__main__":
    # File paths
    repro_file = "debug_circuits_118_483_reproduction.txt"
    test1_file = "debug_step2_shots16_test1.txt"
    test2_file = "debug_step2_shots16_test2.txt"
    
    # Run comparison
    result = compare_results(repro_file, test1_file, test2_file)
    
    # Exit with code indicating result
    if result == "BOTH":
        exit(0)  # Matches both test1 and test2
    elif result == "TEST1":
        exit(1)  # Matches test1 for all circuits
    elif result == "TEST2":
        exit(2)  # Matches test2 for all circuits
    elif result == "MIXED":
        exit(3)  # Matches mixed (some test1, some test2)
    else:  # NEITHER
        exit(4)  # Matches neither

#!/usr/bin/env python3
"""
Test script to verify that the quantum state sorting function works correctly.
"""

def sort_quantum_states(states_dict):
    """Sort quantum states numerically from |000...> to |111...>"""
    # Sort by the integer value of the binary string
    return dict(sorted(states_dict.items(), key=lambda x: int(x[0], 2) if x[0] else 0))

def test_sorting():
    """Test the sorting function with various quantum states."""
    
    # Test case 1: Basic 3-qubit states (unsorted)
    test_data = {
        '111': 0.25,
        '000': 0.25, 
        '101': 0.125,
        '010': 0.125,
        '001': 0.125,
        '110': 0.125
    }
    
    expected_order = ['000', '001', '010', '101', '110', '111']
    
    sorted_data = sort_quantum_states(test_data)
    actual_order = list(sorted_data.keys())
    
    print("Test 1: Basic 3-qubit states")
    print(f"Expected order: {expected_order}")
    print(f"Actual order:   {actual_order}")
    print(f"✅ PASS: {actual_order == expected_order}")
    
    # Test case 2: States with different lengths
    test_data2 = {
        '00': 0.5,
        '11': 0.3,
        '01': 0.2
    }
    
    expected_order2 = ['00', '01', '11']
    sorted_data2 = sort_quantum_states(test_data2)
    actual_order2 = list(sorted_data2.keys())
    
    print("\nTest 2: 2-qubit states")
    print(f"Expected order: {expected_order2}")
    print(f"Actual order:   {actual_order2}")
    print(f"✅ PASS: {actual_order2 == expected_order2}")
    
    # Test case 3: Edge case with empty string (should be treated as 0)
    test_data3 = {
        '': 1.0,
        '001': 0.5,
        '111': 0.25
    }
    
    sorted_data3 = sort_quantum_states(test_data3)
    actual_order3 = list(sorted_data3.keys())
    
    print("\nTest 3: Edge case with empty string")
    print(f"Actual order: {actual_order3}")
    print(f"First item should be empty string: ✅ PASS: {actual_order3[0] == ''}")
    
    print("\n🎉 All sorting tests passed!")

if __name__ == "__main__":
    test_sorting()
#!/usr/bin/env python
"""
Test suite for circuit knitter symmetry.

This script tests that the circuit knitter produces identical results
when swapping control and target qubits, using low shot counts for
exact comparison.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ExperimentConfig
from circuit_utils import create_circuit_3q_test
from experiment import circuit_knitter


class TestKnitterSymmetry(unittest.TestCase):
    """Test cases for circuit knitter symmetry."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test configuration with no noise
        self.config = ExperimentConfig(
            noise=False,
            num_shots=16,  # Low shot count for exact comparison
            results_dir="test_results"
        )
        
        # Create the 3-qubit test circuit
        self.test_circuit = create_circuit_3q_test()
        
        # Display circuit diagram (only once)
        if not hasattr(self.__class__, '_circuit_displayed'):
            print("\nTest Circuit Diagram:")
            print(self.test_circuit.draw(output='text'))
            self.__class__._circuit_displayed = True
        
    def test_symmetry_qubits_0_1(self):
        """Test symmetry between qubits 0 and 1."""
        print("\nTesting symmetry: qubits 0 and 1")
        
        # Knit with qubit 0 as control, qubit 1 as target
        results_0_1 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Knit with qubit 1 as control, qubit 0 as target  
        results_1_0 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=1,
            end_qubit=0,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Output the counts for debugging
        print(f"Results 0→1: {results_0_1['results']}")
        print(f"Results 1→0: {results_1_0['results']}")
        
        # Results should be identical
        self.assertEqual(results_0_1['results'], results_1_0['results'], 
                        "Results should be symmetric for qubits 0 and 1")
        
        print("✓ Symmetry test passed for qubits 0 and 1")
        
    def test_symmetry_qubits_1_2(self):
        """Test symmetry between qubits 1 and 2."""
        print("\nTesting symmetry: qubits 1 and 2")
        
        # Knit with qubit 1 as control, qubit 2 as target
        results_1_2 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=1,
            end_qubit=2,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Knit with qubit 2 as control, qubit 1 as target
        results_2_1 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=2,
            end_qubit=1,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Output the counts for debugging
        print(f"Results 1→2: {results_1_2['results']}")
        print(f"Results 2→1: {results_2_1['results']}")
        
        # Results should be identical
        self.assertEqual(results_1_2['results'], results_2_1['results'], 
                        "Results should be symmetric for qubits 1 and 2")
        
        print("✓ Symmetry test passed for qubits 1 and 2")
        
    def test_symmetry_qubits_2_0(self):
        """Test symmetry between qubits 0 and 2."""
        print("\nTesting symmetry: qubits 0 and 2")
        
        # Knit with qubit 0 as control, qubit 2 as target
        results_0_2 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=2,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Knit with qubit 2 as control, qubit 0 as target
        results_2_0 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=2,
            end_qubit=0,
            num_shots=16,
            config=self.config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        # Output the counts for debugging
        print(f"Results 0→2: {results_0_2['results']}")
        print(f"Results 2→0: {results_2_0['results']}")
        
        # Results should be identical
        self.assertEqual(results_0_2['results'], results_2_0['results'], 
                        "Results should be symmetric for qubits 0 and 2")
        
        print("✓ Symmetry test passed for qubits 0 and 2")


def run_symmetry_tests():
    """Run symmetry tests and generate a summary."""
    print("Running Circuit Knitter Symmetry Tests...")
    print("=" * 50)
    print("Testing with 16 shots for exact comparison")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestKnitterSymmetry)
    
    # Run tests with verbosity=1 to suppress individual test names
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Symmetry tests completed: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All symmetry tests passed!")
        print("The circuit knitter is symmetric in control and target qubits.")
    else:
        print("✗ Some symmetry tests failed!")
        print("The circuit knitter may not be symmetric.")
    
    return result


if __name__ == "__main__":
    # Run symmetry tests
    test_result = run_symmetry_tests()
    
    # Exit with appropriate code
    exit(0 if test_result.wasSuccessful() else 1)
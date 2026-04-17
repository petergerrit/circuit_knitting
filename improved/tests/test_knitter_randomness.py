#!/usr/bin/env python
"""
Test suite for circuit knitter randomness.

This script tests that the circuit knitter produces different results with different seeds
and identical results with the same seed, using the original test circuit.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig
from circuit_utils import create_circuit_3q_test
from experiment import circuit_knitter


class TestKnitterRandomness(unittest.TestCase):
    """Test cases for circuit knitter randomness."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test configuration with no noise
        self.config_no_noise = ExperimentConfig(
            noise=False,
            num_shots=1024,  # Standard shots for statistical significance
            results_dir="test_results"
        )
        

        
        # Use the original test circuit with random unitaries
        self.test_circuit = create_circuit_3q_test()
        
    def test_knitter_randomness(self):
        """Test that circuit knitter produces different results with different seeds and same with same seed."""
        print("\nTesting circuit knitter randomness (no noise)...")
        print("Using 1024 shots with the original test circuit")
        print("Knitting qubits: control=0, target=1")
        
        # Display the test circuit
        print("\nTest Circuit Diagram:")
        print(self.test_circuit.draw(fold=-1))
        
        # First evaluation with seed 42
        seed1 = 42
        print(f"\nEvaluation 1 with seed {seed1}")
        results1 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_no_noise.num_shots,
            config=self.config_no_noise,
            simulator_seed=seed1,
            transpiler_seed=seed1
        )['results']
        print(f"Results 1: {results1}")
        
        # Second evaluation with different seed (123)
        seed2 = 123
        print(f"\nEvaluation 2 with seed {seed2}")
        results2 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_no_noise.num_shots,
            config=self.config_no_noise,
            simulator_seed=seed2,
            transpiler_seed=seed2
        )['results']
        print(f"Results 2: {results2}")
        
        # Third evaluation with original seed (42) - should match results1
        print(f"\nEvaluation 3 with seed {seed1} (same as evaluation 1)")
        results3 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_no_noise.num_shots,
            config=self.config_no_noise,
            simulator_seed=seed1,
            transpiler_seed=seed1
        )['results']
        print(f"Results 3: {results3}")
        
        # Verify reproducibility: results1 should equal results3
        self.assertEqual(results1, results3, 
                        "Same seed should produce identical results")
        print("✓ Reproducibility confirmed: results1 == results3")
        
        # Verify different seeds produce different results
        self.assertNotEqual(results1, results2, 
                           "Different seeds should produce different results")
        print("✓ Randomness confirmed: results1 != results2")
        
        # Additional check: results3 should also differ from results2
        self.assertNotEqual(results3, results2, 
                           "Different seeds should produce different results (3 vs 2)")
        print("✓ Randomness confirmed: results3 != results2")
        
        print("\n✓ All circuit knitter randomness tests passed (no noise)!")
    



def run_knitter_randomness_tests():
    """Run circuit knitter randomness tests and generate a summary."""
    print("Running Circuit Knitter Randomness Tests...")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestKnitterRandomness)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 55)
    print(f"Circuit knitter randomness tests completed: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ Circuit knitter randomness verified!")
        print("  - Same seeds produce identical results")
        print("  - Different seeds produce different results")
    else:
        print("✗ Circuit knitter randomness tests failed!")
    
    return result


if __name__ == "__main__":
    # Run circuit knitter randomness tests
    test_result = run_knitter_randomness_tests()
    
    # Exit with appropriate code
    exit(0 if test_result.wasSuccessful() else 1)
#!/usr/bin/env python
"""
Test for circuit knitter noise comparison using a simple circuit with CNOT.

This test uses a simple 2-qubit circuit with Hadamard gates, a CNOT gate, and measurements
to test the circuit knitter's behavior with and without noise.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from config import ExperimentConfig
from experiment import circuit_knitter


class TestKnitterNoise(unittest.TestCase):
    """Test cases for circuit knitter noise comparison."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configurations for no noise and with noise
        self.config_no_noise = ExperimentConfig(
            noise=False,
            num_shots=8,  # Low shots for testing
            results_dir="test_results"
        )
        
        self.config_with_noise = ExperimentConfig(
            noise=True,
            num_shots=8,  # Very low shots for noisy testing (slow backend)
            results_dir="test_results",
            optimization_level=1  # Match knitter's optimization level
        )
        
        # Create a simple circuit that the knitter can work with
        # Based on the minimal circuit from test_minimal_noise.py but with CNOT
        self.test_circuit = QuantumCircuit(2, 2)
        self.test_circuit.h(0)  # Hadamard gate creates superposition
        self.test_circuit.h(1)
        self.test_circuit.cx(0, 1)  # CNOT gate (required for knitter)
        self.test_circuit.measure([0, 1], [0, 1])
    
    def test_knitter_simple_circuit(self):
        """Test circuit knitter with simple circuit using 4 evaluations."""
        print("\nTest Circuit Diagram:")
        print(self.test_circuit.draw(output='text'))
        print("\nTesting circuit knitter with noise comparison...")
        print("Using 8 shots for all evaluations (slow backend)")
        print("Knitting qubits: control=0, target=1")
        
        # Evaluation 1: Noisy with seed 42
        print(f"\nEvaluation 1 (noisy, seed 42)")
        results1 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_with_noise.num_shots,
            config=self.config_with_noise,
            simulator_seed=42,
            transpiler_seed=42
        )['results']
        print(f"Results 1: {results1}")
        
        # Evaluation 2: Noisy with seed 123
        print(f"\nEvaluation 2 (noisy, seed 123)")
        results2 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_with_noise.num_shots,
            config=self.config_with_noise,
            simulator_seed=123,
            transpiler_seed=123
        )['results']
        print(f"Results 2: {results2}")
        
        # Evaluation 3: Noisy with seed 42 (should match results1)
        print(f"\nEvaluation 3 (noisy, seed 42)")
        results3 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_with_noise.num_shots,
            config=self.config_with_noise,
            simulator_seed=42,
            transpiler_seed=42
        )['results']
        print(f"Results 3: {results3}")
        
        # Evaluation 4: Noiseless with seed 42
        print(f"\nEvaluation 4 (noiseless, seed 42)")
        results4 = circuit_knitter(
            circuit=self.test_circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=self.config_no_noise.num_shots,
            config=self.config_no_noise,
            simulator_seed=42,
            transpiler_seed=42
        )['results']
        print(f"Results 4: {results4}")
        
        # Verify reproducibility: results1 should equal results3
        self.assertEqual(results1, results3, 
                        "Same seed with noise should produce identical results")
        print("✓ Reproducibility confirmed: results1 == results3")
        
        # Verify different seeds produce different results
        self.assertNotEqual(results1, results2, 
                           "Different seeds with noise should produce different results")
        print("✓ Randomness confirmed: results1 != results2")
        
        # Verify noise makes a difference
        self.assertNotEqual(results1, results4, 
                           "Noisy and noiseless results should be different")
        print("✓ Noise effect confirmed: results1 != results4")
        
        print("\n✓ All circuit knitter noise comparison tests passed!")


def run_knitter_noise_tests():
    """Run circuit knitter noise tests and generate a summary."""
    print("Running Circuit Knitter Noise Comparison Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestKnitterNoise)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Circuit knitter noise tests completed: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ Circuit knitter noise comparison verified!")
        print("  - Same seeds with noise produce identical results")
        print("  - Different seeds with noise produce different results")
        print("  - Noisy and noiseless results are different")
        print("  - Using simple circuit with CNOT (2 qubits, Hadamard + CNOT + measure)")
    else:
        print("✗ Circuit knitter noise comparison tests failed!")
    
    return result


if __name__ == "__main__":
    # Run circuit knitter noise tests
    test_result = run_knitter_noise_tests()
    
    # Exit with appropriate code
    exit(0 if test_result.wasSuccessful() else 1)
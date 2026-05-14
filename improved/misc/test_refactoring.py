#!/usr/bin/env python
"""
Test script to verify the refactoring works correctly.
"""

import sys
import os
sys.path.insert(0, '.')

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test circuit imports
        from circuits.basic_circuits import create_circuit_3q_test
        print("✓ circuits.basic_circuits import successful")
        
        # Test knitter imports
        from knitter.knitter import circuit_knitter
        from knitter.execution import run_circuit_experiment, my_measure
        print("✓ knitter imports successful")
        
        # Test evaluation imports
        from evaluation.evaluator import evaluate_with_knitting, evaluate_without_knitting
        print("✓ evaluation imports successful")
        
        # Test config import
        from config import ExperimentConfig
        print("✓ config import successful")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test that basic functionality works."""
    print("\nTesting basic functionality...")
    
    try:
        from config import ExperimentConfig
        from circuits.basic_circuits import create_circuit_3q_test
        from knitter.knitter import circuit_knitter
        
        # Create a simple config
        config = ExperimentConfig(
            noise=False,
            num_shots=16,
            results_dir="test_results"
        )
        
        # Create a test circuit
        circuit = create_circuit_3q_test()
        print(f"✓ Created test circuit with {circuit.num_qubits} qubits")
        
        # Test knitter (this will be quick with low shots)
        result = circuit_knitter(
            circuit=circuit,
            start_qubit=0,
            end_qubit=1,
            num_shots=16,
            config=config,
            simulator_seed=42,
            transpiler_seed=42
        )
        
        print(f"✓ Knitter executed successfully, got {len(result['results'])} result entries")
        print(f"✓ Results: {result['results']}")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing refactored code structure...")
    print("=" * 50)
    
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("🎉 Refactoring successful! All tests passed.")
        sys.exit(0)
    else:
        print("❌ Refactoring has issues. Please check the errors above.")
        sys.exit(1)
#!/usr/bin/env python3
"""Display circuit diagrams and run test circuits."""

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from circuit_utils import create_circuit_3q_test, create_circuit_2q_test
from statevector_analysis import run_statevector_analysis

def show_circuit(circuit, title):
    """Display circuit diagram."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(circuit)
    
    # Also save as text diagram
    try:
        diagram = circuit.draw(output='text')
        print(diagram)
        with open(f"{title.replace(' ', '_').lower()}.txt", 'w') as f:
            f.write(str(circuit) + "\n\n")
            f.write(str(diagram))
        print(f"✓ Saved text diagram to {title.replace(' ', '_').lower()}.txt")
    except Exception as e:
        print(f"Note: Could not generate text diagram: {e}")

def run_circuit(circuit, title, shots=1024):
    """Run circuit and show measurement outcomes using statevector analysis."""
    print(f"\n{'='*60}")
    print(f"Running {title} - {shots} shots")
    print('='*60)
    
    # Use statevector analysis module
    results = run_statevector_analysis(circuit, shots)
    
    print(f"Measurement outcomes (theoretical probabilities):")
    for state, prob in results['theoretical_ratios'].items():
        count = results['counts'][state]
        print(f"  {state}: {count} ({prob*100:.1f}%)")
    
    return results['counts']

if __name__ == "__main__":
    # Create circuits
    qc_3qubit = create_circuit_3q_test()
    qc_2qubit = create_circuit_2q_test()
    
    # Show diagrams
    show_circuit(qc_3qubit, "3-Qubit Test Circuit")
    show_circuit(qc_2qubit, "2-Qubit Test Circuit")
    
    # Run circuits
    counts_3q = run_circuit(qc_3qubit, "3-Qubit Test Circuit")
    counts_2q = run_circuit(qc_2qubit, "2-Qubit Test Circuit")
    
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"3-qubit circuit: {len(qc_3qubit.count_ops())} gate types, {sum(qc_3qubit.count_ops().values())} total operations")
    print(f"2-qubit circuit: {len(qc_2qubit.count_ops())} gate types, {sum(qc_2qubit.count_ops().values())} total operations")
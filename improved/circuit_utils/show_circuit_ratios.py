#!/usr/bin/env python3
"""Display circuit diagrams and run test circuits, showing results as ratios."""

from qiskit import QuantumCircuit
import numpy as np
from circuit_utils import create_circuit_3q_test, create_circuit_2q_test
from statevector_analysis import run_statevector_analysis

def show_circuit(circuit, title):
    """Display circuit diagram."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(circuit)
    
    # Save as text diagram
    try:
        diagram = circuit.draw(output='text')
        print(diagram)
    except Exception as e:
        print(f"Note: Could not generate text diagram: {e}")

def run_circuit_with_ratios(circuit, title, shots=1024):
    """Run circuit and show measurement outcomes as ratios."""
    print(f"\n{'='*60}")
    print(f"Running {title} - {shots} shots")
    print('='*60)
    
    # Remove measurements for statevector simulation
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    statevector = Statevector.from_instruction(circuit_no_meas)
    probs = statevector.probabilities()
    
    # Convert probabilities to counts
    counts = {}
    n_qubits = circuit_no_meas.num_qubits
    for i, prob in enumerate(probs):
        if prob > 0:
            state = format(i, f'0{n_qubits}b')
            count = int(prob * shots)
            counts[state] = count
    
    # Calculate ratios
    total = sum(counts.values())
    print(f"Total counts: {total}")
    print(f"Measurement outcome ratios:")
    for state, count in sorted(counts.items()):
        ratio = count / total
        print(f"  {state}: {ratio:.6f} ({count} counts)")
    
    return counts

if __name__ == "__main__":
    # Create circuits
    qc_3qubit = create_circuit_3q_test()
    qc_2qubit = create_circuit_2q_test()
    
    # Show diagrams
    show_circuit(qc_3qubit, "3-Qubit Test Circuit")
    show_circuit(qc_2qubit, "2-Qubit Test Circuit")
    
    # Run circuits with ratios
    counts_3q = run_circuit_with_ratios(qc_3qubit, "3-Qubit Test Circuit")
    counts_2q = run_circuit_with_ratios(qc_2qubit, "2-Qubit Test Circuit")
    
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"3-qubit circuit: {len(qc_3qubit.count_ops())} gate types, {sum(qc_3qubit.count_ops().values())} total operations")
    print(f"2-qubit circuit: {len(qc_2qubit.count_ops())} gate types, {sum(qc_2qubit.count_ops().values())} total operations")
    shots_used = 1024  # Default shots per circuit
    print(f"Total shots used: {shots_used * 2}")
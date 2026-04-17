"""
Circuit utility functions for quantum circuit operations.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from typing import Optional


def create_circuit_3q_test() -> QuantumCircuit:
    """Create the 3-qubit test circuit for experimentation."""
    rand_init = random_unitary(8, seed=3)
    rand_1 = random_unitary(2, seed=1)
    rand_2 = random_unitary(2, seed=2)
    rand_3 = random_unitary(2, seed=3)
    rand_4 = random_unitary(2, seed=4)
    rand_5 = random_unitary(2, seed=5)
    rand_6 = random_unitary(2, seed=6)

    qc = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement

    qc.unitary(rand_init, range(3), label='rand_init')
    qc.cx(0, 1)
    qc.cx(2, 0)
    qc.unitary(rand_1, 1, label='rand_1')
    qc.unitary(rand_2, 2, label='rand_2')
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.unitary(rand_3, 0, label='rand_3')
    qc.unitary(rand_4, 2, label='rand_4')
    qc.cx(1, 0)
    qc.cx(2, 0)
    qc.cx(2, 1)
    qc.unitary(rand_5, 0, label='rand_5')
    qc.unitary(rand_6, 1, label='rand_6')
    
    # Add measurements to all qubits
    qc.measure([0, 1, 2], [0, 1, 2])

    # Return the quantum circuit directly, not as a gate
    return qc


def trotter_stepper(step: int, n_qubits: int, epsilon: float, mass: float, mid: int) -> QuantumCircuit:
    """Create a Trotter step circuit."""
    # Implementation would go here
    # This is a placeholder for the actual trotter step implementation
    qc = QuantumCircuit(n_qubits)
    # Add actual trotter step gates based on parameters
    return qc


def test_circuit_1() -> QuantumCircuit:
    """Create the 2-qubit test circuit from the parent directory."""
    rand_init = random_unitary(4, seed=3)
    rand_1 = random_unitary(2, seed=1)
    rand_2 = random_unitary(2, seed=2)
    
    qc = QuantumCircuit(2, 0)
    
    qc.unitary(rand_init, range(2), label='rand_init')
    qc.cx(0, 1)
    qc.unitary(rand_1, 0, label='rand_1')
    qc.unitary(rand_2, 1, label='rand_2')
    
    # Return the circuit directly (not as a gate like in the original)
    return qc


def create_circuit_2q_test() -> QuantumCircuit:
    """Create the 2-qubit test circuit with measurements."""
    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
    
    # Add the test circuit
    test_circuit = test_circuit_1()
    qc.append(test_circuit, range(2))
    qc.decompose()
    qc.measure_all()
    qc = qc.decompose()
    
    return qc


def prepare_circuit_for_execution(circuit: QuantumCircuit) -> QuantumCircuit:
    """Prepare a circuit for execution by decomposing and adding measurements."""
    prepared_circuit = circuit.decompose().decompose()
    prepared_circuit.measure_all()
    return prepared_circuit
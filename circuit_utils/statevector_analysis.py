#!/usr/bin/env python3
"""
Statevector analysis module for quantum circuits.

This module provides functions for performing ideal statevector simulations
to determine the theoretical probabilities of quantum circuit outcomes.
"""

from qiskit.quantum_info import Statevector
import numpy as np


def run_statevector_analysis(circuit, shots=1024):
    """
    Run ideal statevector simulation and return theoretical probabilities.
    
    This function performs a noiseless, ideal quantum simulation to determine
    the exact probabilities of all possible measurement outcomes.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to analyze
        shots (int): Number of shots to simulate (for count conversion)
        
    Returns:
        dict: Dictionary containing:
            - 'probabilities': Array of outcome probabilities
            - 'counts': Dictionary of outcome counts
            - 'statevector': The computed statevector
            - 'theoretical_ratios': Dictionary of outcome ratios
    """
    # Remove measurements for statevector simulation
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    statevector = Statevector.from_instruction(circuit_no_meas)
    probs = statevector.probabilities()
    
    # Convert probabilities to counts and ratios
    counts = {}
    theoretical_ratios = {}
    n_qubits = circuit_no_meas.num_qubits
    
    for i, prob in enumerate(probs):
        if prob > 0:
            state = format(i, f'0{n_qubits}b')
            count = int(prob * shots)
            counts[state] = count
            theoretical_ratios[state] = prob
    
    return {
        'probabilities': probs,
        'counts': counts,
        'statevector': statevector,
        'theoretical_ratios': theoretical_ratios
    }


def get_theoretical_probabilities(circuit):
    """
    Get theoretical probabilities for a quantum circuit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to analyze
        
    Returns:
        dict: Mapping of outcome states to their theoretical probabilities
    """
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    statevector = Statevector.from_instruction(circuit_no_meas)
    probs = statevector.probabilities()
    
    theoretical_probs = {}
    n_qubits = circuit_no_meas.num_qubits
    
    for i, prob in enumerate(probs):
        if prob > 0:
            state = format(i, f'0{n_qubits}b')
            theoretical_probs[state] = prob
    
    return theoretical_probs
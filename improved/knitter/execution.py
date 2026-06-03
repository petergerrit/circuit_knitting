"""
Execution helpers for the circuit knitter.

This module contains low-level functions for circuit execution, measurement,
and result processing that support the main knitting algorithm.
"""
import numpy as np
from typing import Dict, List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2
from collections import defaultdict


def my_measure(
    circuit_data: List,
    conq: int,
    tarq: int,
    num_qubits: int,
    num_cx: int,
    num_shots: int,
    simulator_seed: int,
    transpiler_seed: int,
    noise: bool,
    return_memory: bool = False
) -> Dict[str, int]:
    """
    Assembles a quantum circuit from a list of circuit components, runs and measures
    the circuit and returns the outcomes.
    
    Args:
        circuit_data: List of circuit components
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_qubits: Number of qubits in the original circuit
        num_cx: Number of CNOTs in the original circuit
        num_shots: Number of shots
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        noise: Whether to use noisy backend
        return_memory: If True, return dict with both 'counts' and 'memory' (raw bitstrings)
        
    Returns:
        If return_memory=False: Dictionary of measurement counts (includes internal measurements)
        If return_memory=True: Dictionary with 'counts' and 'memory' (list of bitstrings in order)
    """
    # Initialize circuit and assemble it from components in circuit_data
    my_circ = QuantumCircuit(num_qubits, num_cx+num_qubits)
    for item in circuit_data:
        my_circ.append(item)
    my_circ.measure([*range(num_qubits)], [*range(num_cx, num_cx + num_qubits)])

    # Select noisy or ideal backend
    if noise:
        backend = FakeWashingtonV2()
    else:
        backend = AerSimulator()
    
    # Transpiled circuit
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=transpiler_seed)
    transpiled_circuit = pass_manager.run(my_circ)
    
    # Initialize sampler
    options = {"simulator": {"seed_simulator": simulator_seed}}
    sampler = SamplerV2(backend, options=options)
    
    # Run job and get results
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0]
    
    if return_memory:
        counts = result.data.c.get_counts()
        memory = result.data.c.get_memory()
        return {'counts': counts, 'memory': memory}
    else:
        return result.data.c.get_counts()


def comb_measure(result_in: Dict[str, int], conq: int, tarq: int, num_cx: int) -> Dict[str, int]:
    """
    Combines quiskit results with circuit internal measurements.
    
    Args:
        result_in: Dictionary of measurement outcomes with internal measurements
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_cx: Number of CNOTs in the original circuit
        
    Returns:
        Dictionary of combined measurement outcomes
    """
    result_dict = defaultdict(int)
    for item in result_in:
        # Note: Use len(item)-num_cx instead of -num_cx to handle num_cx=0 case
        # When num_cx=0, item[:-0] returns empty string, but item[:len(item)] returns full string
        end_meas = item[:len(item)-num_cx]
        int_meas = item[len(item)-num_cx:]
        if end_meas[-conq-1]+end_meas[-tarq-1] == '00':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        elif end_meas[-conq-1]+end_meas[-tarq-1] == '10':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        elif end_meas[-conq-1]+end_meas[-tarq-1] == '01':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        else:
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
    return dict(result_dict)


def knit_lister(circuit: QuantumCircuit, conq: int, tarq: int, meas: int, num_cx: int) -> List:
    """
    Assembles the list of circuit components in the knitting decomposition.
    
    Args:
        circuit: The circuit to be knitted
        conq: Index of the control qubit
        tarq: Index of the target qubit
        meas: Which CNOT is being knitted
        num_cx: Total number of CNOTs in the circuit
        
    Returns:
        List of circuit components for the knitting decomposition (length 6)
    
    DEBUG ONLY: The latter 5 circuit components are commented out below to speed up
    execution. This breaks the mathematical correctness of circuit knitting but is
    useful for debugging execution flow. Uncomment all 6 components for production use.
    """
    # DEBUG: Commented out to speed up execution - uncomment for production
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    circuit_list =[]
    qc.rz(np.pi/2, tarq)
    qc.rx(np.pi/2, conq)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(-np.pi/2, tarq)
    qc.rx(-np.pi/2, conq)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarq, meas)
    qc.rx(np.pi, conq)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarq, meas)
    circuit_list.append(qc[:])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(np.pi, tarq)
    qc.h(conq)
    qc.measure(conq, meas)
    qc.h(conq)
    circuit_list.append(qc[:4])
    del(qc)
    # qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    # qc.h(conq)
    # qc.measure(conq, meas)
    # qc.h(conq)
    # circuit_list.append(qc[:3])
    # del(qc)
    return circuit_list


def flattener(nested_list: List) -> List:
    """
    Flatten a nested list structure.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flattener(item))
        else:
            flat_list.append(item)
    return flat_list

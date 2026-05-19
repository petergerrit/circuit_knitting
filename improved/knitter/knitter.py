"""
Circuit knitter implementation.

This module contains the main circuit knitting algorithm that decomposes
quantum circuits containing CNOT gates into knitted subcircuits.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from itertools import product
from collections import defaultdict

# Import execution helpers
from .execution import my_measure, comb_measure, knit_lister, flattener

try:
    from .config import ExperimentConfig
    from circuits.basic_circuits import prepare_circuit_for_execution
except ImportError:
    from config import ExperimentConfig
    from circuits.basic_circuits import prepare_circuit_for_execution


def fully_decompose_circuit(circuit: 'QuantumCircuit', max_iterations: int = 10, preserve_unitary: bool = True) -> 'QuantumCircuit':
    """
    Recursively decompose a quantum circuit until all gates are basis gates.
    
    Args:
        circuit: Input quantum circuit potentially containing composite gates
        max_iterations: Maximum number of decomposition iterations (safety limit)
        preserve_unitary: If True, keep unitary gates atomic (don't decompose them)
        
    Returns:
        Fully decomposed quantum circuit with only basis gates
    """
    # List of Qiskit basis gate names
    basis_gates = {'cx', 'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 
                   'rx', 'ry', 'rz', 'sx', 'sxdg', 'swap',
                   'measure', 'barrier', 'reset', 'id', 'u1', 'u2', 'u3',
                   'p', 'cp', 'ccx', 'mcx', 'r', 'rxx', 'ryy', 'rzz', 'rzx'}
    
    if preserve_unitary:
        basis_gates = basis_gates | {'unitary'}
    
    for _ in range(max_iterations):
        # Find gates that need decomposition (not in basis_gates)
        gates_to_decompose = set()
        for gate in circuit.data:
            if gate.operation.name not in basis_gates:
                gates_to_decompose.add(gate.operation.name)
        
        if not gates_to_decompose:
            return circuit
        
        # Decompose only the non-basis gates
        decomposed = circuit.decompose(gates_to_decompose=list(gates_to_decompose))
        
        # Check if all gates are now primitive
        all_primitive = True
        for gate in decomposed.data:
            if gate.operation.name not in basis_gates:
                all_primitive = False
                break
        if all_primitive:
            return decomposed
        circuit = decomposed
    
    # If we reach max iterations, return the best effort
    return circuit


def circuit_knitter(
    circuit: 'QuantumCircuit',
    start_qubit: int,
    end_qubit: int,
    num_shots: int,
    config: ExperimentConfig,
    simulator_seed: Optional[int] = None,
    transpiler_seed: Optional[int] = None,
    decompose: bool = True
) -> Dict[str, Any]:
    """
    Perform circuit knitting on a given circuit using the full knitting algorithm.
    
    Args:
        circuit: Input quantum circuit
        start_qubit: Starting qubit index (control qubit)
        end_qubit: Ending qubit index (target qubit)
        num_shots: Number of shots for measurement
        config: Experiment configuration
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        decompose: Whether to fully decompose the circuit (default True)
        
    Returns:
        Dictionary containing knitting results and metadata
    """
    # Fully decompose the circuit to get individual gates
    # Note: preserve_unitary=True prevents decomposing unitary gates (treats them as atomic)
    if decompose:
        circuit = fully_decompose_circuit(circuit, preserve_unitary=True)
    
    conq = start_qubit
    tarq = end_qubit
    
    # List of positions in circuit of CNOT gates acting on control and target qubits
    cx_list = [i for i, gate in enumerate(circuit.data) if gate.operation.name == 'cx' and \
               ((circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (tarq, conq) or \
               (circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (conq, tarq))]
    num_cx = len(cx_list)
    
    # List of 6 numerical prefactors of knitting terms -> list of 6**n prefactors for n CNOTS 
    prefac_list = [1/2, 1/2, -1/2, 1/2, -1/2, 1/2]
    prefac_list = [float(np.prod(elem)) for elem in [*product(*[prefac_list for i in range(num_cx)])]]
    
    # Creates a list corresponding to the circuit to be knitted with each target CNOT replaced by a list of the knitting terms
    nest_list = []
    current_cx = 0
    
    for i, gate in enumerate(circuit.data):
        if gate.operation.name == 'cx':
            if (gate.qubits[0]._index, gate.qubits[1]._index) == (tarq, conq):
                nest_list.append(knit_lister(circuit, conq, tarq, current_cx, num_cx))
                current_cx += 1
            elif (gate.qubits[0]._index, gate.qubits[1]._index) == (conq, tarq):
                nest_list.append(knit_lister(circuit, tarq, conq, current_cx, num_cx))
                current_cx += 1
            else:
                nest_list.append([gate])
        elif gate.operation.name not in ("measure", "barrier"):
            nest_list.append([gate])
        else:
            None
    
    # A list of length 6**n containing the instructions to assemble the terms of the decomposition
    circuits = [*product(*nest_list)]
    circuits = [flattener(item) for item in circuits]
    
    # Execute all knitting terms and combine results
    cum_tot = defaultdict(int)
    for i, item in enumerate(circuits):
        # Use provided seeds for reproducibility, or generate random seeds for variability
        if simulator_seed is not None:
            current_simulator_seed = simulator_seed + i  # Add iteration to ensure different seeds for each circuit
        else:
            current_simulator_seed = np.random.randint(1024**2)
            
        if transpiler_seed is not None:
            current_transpiler_seed = transpiler_seed + i  # Add iteration to ensure different seeds for each circuit
        else:
            current_transpiler_seed = np.random.randint(1024**2)
        
        temp_res_internal_meas = my_measure(item, conq, tarq, circuit.num_qubits, num_cx, num_shots, 
                                            current_simulator_seed, current_transpiler_seed, config.noise)
        temp_res = comb_measure(temp_res_internal_meas, conq, tarq, num_cx)
        for sub_item in temp_res:
            cum_tot[sub_item] += temp_res[sub_item] * prefac_list[i]
    
    res_dict = {key: cum_tot[key] for key in sorted(cum_tot)}
    
    return {
        'circuit': circuit,
        'results': res_dict,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat(),
        'qubits': {'start_qubit': start_qubit, 'end_qubit': end_qubit},
        'num_cnots': num_cx
    }


def save_experiment_results(
    results: Dict[str, Any],
    filename: str,
    config: ExperimentConfig
) -> str:
    """
    Save experiment results to a file.
    
    Args:
        results: Experiment results to save
        filename: Base filename for results
        config: Experiment configuration
        
    Returns:
        Full path to saved file
    """
    import pickle
    
    # Create full path
    full_path = f"{config.results_dir}/{filename}.pkl"
    
    # Save results
    with open(full_path, 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return full_path


def load_experiment_results(filename: str, config: ExperimentConfig) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        filename: Base filename of results
        config: Experiment configuration
        
    Returns:
        Loaded experiment results
    """
    import pickle
    
    full_path = f"{config.results_dir}/{filename}.pkl"
    
    with open(full_path, 'rb') as file:
        return pickle.load(file)

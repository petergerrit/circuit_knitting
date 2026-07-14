"""Utility functions for circuit simulation and statistics."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2
from collections import defaultdict
from itertools import product


def do_run(circuit, num_shots, noise=False, simulator_seed=1, transpiler_seed=1):
    """Run a circuit and return measurement counts.
    
    Parameters:
        circuit: QuantumCircuit to run
        num_shots: Number of shots
        noise: Whether to use noisy backend
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
    
    Returns:
        dict: Measurement counts with sorted keys
    """
    if noise:
        backend = FakeWashingtonV2()
    else:
        backend = AerSimulator()
    
    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=transpiler_seed
    )
    transpiled_circuit = pass_manager.run(circuit)
    
    options = {"simulator": {"seed_simulator": simulator_seed}}
    sampler = SamplerV2(backend, options=options)
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result = job.result()[0].data.meas.get_counts()
    return {key: result[key] for key in sorted(result)}


def fermion_number(counts, mid):
    """Calculate mean fermion number from measurement counts.
    
    Parameters:
        counts: Dictionary of measurement outcomes and counts
        mid: Index of the middle qubit
    
    Returns:
        float: Mean fermion number
    """
    mean = 0
    total_counts = sum(counts.values())
    for s in counts:
        p = s[mid+1]
        if p == '1':
            mean += 1./total_counts * counts[s]
    return mean


def bootstrap_error(counts, mid, shots, seed=1):
    """Calculate bootstrap error for fermion number.
    
    Parameters:
        counts: Dictionary of measurement outcomes and counts
        mid: Index of the middle qubit
        shots: Total number of shots
        seed: Random seed for reproducibility
    
    Returns:
        float: Bootstrap error estimate
    """
    np.random.seed(seed)
    nshots = shots
    B = 100
    k = list(counts.keys())
    prob = [np.abs(counts[a]) for a in k]
    means = []
    
    for b in range(B):
        m = 0
        samples = np.random.choice(k, size=nshots, p=(prob / sum(prob)))
        for s in samples:
            p = s[mid+1]
            if p == '1' and counts[s] > 0:
                m += 1./nshots
            elif p == '1':
                m -= 1./nshots
        means.append(m)
    
    return float(np.std(means))


def knit_lister(circuit, conloc, tarloc, meas, num_cx):
    """Assembles the list of circuit components in the knitting decomposition.
    
    Parameters:
        circuit: The circuit to be knitted
        conloc: Index of the control qubit
        tarloc: Index of the target qubit  
        meas: Which of the num_cx CNOTs is being knitted
        num_cx: Total number of CNOTs in the original circuit
    
    Returns:
        list: List of circuit components for knitting decomposition
    """
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    circuit_list = []
    
    qc.rz(np.pi/2, tarloc)
    qc.rx(np.pi/2, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(-np.pi/2, tarloc)
    qc.rx(-np.pi/2, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarloc, meas)
    qc.rx(np.pi, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarloc, meas)
    circuit_list.append(qc[:])
    del(qc)
    
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(np.pi, tarloc)
    qc.h(conloc)
    qc.measure(conloc, meas)
    qc.h(conloc)
    circuit_list.append(qc[:4])
    del(qc)
    
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.h(conloc)
    qc.measure(conloc, meas)
    qc.h(conloc)
    circuit_list.append(qc[:3])
    del(qc)
    
    return circuit_list


def flattener(in_list):
    """Flattens a nested list."""
    out_list = []
    for i in in_list:
        if isinstance(i, list):
            for item in i:
                out_list.append(item)
        else:
            out_list.append(i)
    return out_list


def my_measure(my_list, conq, tarq, num_qbits, num_cx, num_shots, simulator_seed, transpiler_seed, noise=False):
    """Assembles a quantum circuit from a list of circuit components, runs and measures.
    
    Parameters:
        my_list: List of circuit components
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_qbits: Number of qubits in the original circuit
        num_cx: Number of CNOTs in the original circuit
        num_shots: Number of shots
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        noise: Whether to use noisy backend
    
    Returns:
        dict: Measurement counts
    """
    my_circ = QuantumCircuit(num_qbits, num_cx+num_qbits)
    for item in my_list:
        my_circ.append(item)
    my_circ.measure([*range(num_qbits)], [*range(-num_qbits, 0)])

    if noise:
        backend = FakeWashingtonV2()
    else:
        backend = AerSimulator()

    pass_manager = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=transpiler_seed
    )
    transpiled_circuit = pass_manager.run(my_circ)

    options = {"simulator": {"seed_simulator": simulator_seed}}
    sampler = SamplerV2(backend, options=options)

    job = sampler.run([transpiled_circuit], shots=num_shots)
    result_dict = job.result()[0].data.c.get_counts()

    return result_dict


def comb_measure(result_in, conq, tarq, num_cx):
    """Combines quiskit results with circuit internal measurements.
    
    Parameters:
        result_in: Dictionary of measurement outcomes with circuit internal measurements
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_cx: Number of CNOTs in the original circuit
    
    Returns:
        dict: Combined measurement results
    """
    result_dict = defaultdict(int)
    for item in result_in:
        end_meas = item[:-num_cx]
        int_meas = item[-num_cx:]
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
    return result_dict


def circuit_knitter(qc, conq, tarq, num_shots, simulator_seed=1, transpiler_seed=1, noise=False):
    """Perform circuit knitting, measure and return outcomes.
    
    Parameters:
        qc: QuantumCircuit to be knitted
        conq: Index of the control qubit to be knitted
        tarq: Index of the target qubit to be knitted
        num_shots: Number of measurements to perform
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        noise: Whether to use noisy backend
    
    Returns:
        dict: Sorted dictionary of measurement outcomes
    """
    # List of positions in circuit of CNOT gates acting on control and target qubits
    cx_list = [i for i, gate in enumerate(qc.data) 
               if gate.operation.name == 'cx' and \
               ((qc.data[i].qubits[0]._index, qc.data[i].qubits[1]._index) == (tarq, conq) or \
                (qc.data[i].qubits[0]._index, qc.data[i].qubits[1]._index) == (conq, tarq))]
    num_cx = len(cx_list)

    # List of 6 numerical prefactors of knitting terms -> list of 6**n prefactors for n CNOTS
    prefac_list = [1/2, 1/2, -1/2, 1/2, -1/2, 1/2]
    prefac_list = [float(np.prod(elem)) for elem in [*product(*[prefac_list for i in range(num_cx)])]]

    # Creates a list corresponding to the circuit to be knitted with each target CNOT replaced
    nest_list = []
    current_cx = 0
    for i, gate in enumerate(qc.data):
        if gate.operation.name == 'cx':
            if (gate.qubits[0]._index, gate.qubits[1]._index) == (tarq, conq):
                nest_list.append(knit_lister(qc, conq, tarq, current_cx, num_cx))
                current_cx += 1
            elif (gate.qubits[0]._index, gate.qubits[1]._index) == (conq, tarq):
                nest_list.append(knit_lister(qc, tarq, conq, current_cx, num_cx))
                current_cx += 1
            else:
                nest_list.append([gate])
        elif gate.operation.name not in ("measure", "barrier"):
            nest_list.append([gate])

    # A list of length 6**n containing the instructions to assemble the terms of the decomposition
    circuits = [*product(*nest_list)]
    circuits = [flattener(item) for item in circuits]

    cum_tot = defaultdict(int)
    for i, item in enumerate(prefac_list):
        np.random.seed(simulator_seed)
        simulator_seed = np.random.randint(1024**2)
        np.random.seed(transpiler_seed)
        transpiler_seed = np.random.randint(1024**2)
        temp_res_internal_meas = my_measure(circuits[i], conq, tarq, qc.num_qubits, num_cx, 
                                             num_shots, simulator_seed, transpiler_seed, noise)
        temp_res = comb_measure(temp_res_internal_meas, conq, tarq, num_cx)
        for sub_item in temp_res:
            cum_tot[sub_item] += temp_res[sub_item] * item
    
    res_dict = {key: cum_tot[key] for key in sorted(cum_tot)}
    return res_dict

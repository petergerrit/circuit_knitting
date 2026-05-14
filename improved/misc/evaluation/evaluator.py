"""
Circuit knitting evaluator.

This module provides functions to evaluate and compare the performance
of circuit knitting versus direct circuit execution.
"""
from typing import Dict, Any
from knitter.knitter import circuit_knitter
from knitter.execution import run_circuit_experiment


def evaluate_with_knitting(circuit: 'QuantumCircuit', start_qubit: int, end_qubit: int, 
                          config: 'ExperimentConfig', num_shots: int = 1024) -> Dict[str, Any]:
    """
    Evaluate a circuit using the circuit knitter.
    
    Args:
        circuit: Quantum circuit to evaluate
        start_qubit: Control qubit index
        end_qubit: Target qubit index
        config: Experiment configuration
        num_shots: Number of shots
        
    Returns:
        Dictionary containing evaluation results and metadata
    """
    try:
        from config import ExperimentConfig
    except ImportError:
        from knitter.config import ExperimentConfig
    
    # Run circuit through the knitter
    result = circuit_knitter(
        circuit=circuit,
        start_qubit=start_qubit,
        end_qubit=end_qubit,
        num_shots=num_shots,
        config=config,
        simulator_seed=42,
        transpiler_seed=42
    )
    
    # Add evaluation-specific metadata
    result['evaluation_type'] = 'knitted'
    result['circuit_size'] = circuit.num_qubits
    
    return result


def evaluate_without_knitting(circuit: 'QuantumCircuit', config: 'ExperimentConfig', 
                              num_shots: int = 1024) -> Dict[str, Any]:
    """
    Evaluate a circuit using direct execution (no knitting).
    
    Args:
        circuit: Quantum circuit to evaluate
        config: Experiment configuration
        num_shots: Number of shots
        
    Returns:
        Dictionary containing evaluation results and metadata
    """
    try:
        from config import ExperimentConfig
    except ImportError:
        from knitter.config import ExperimentConfig
    
    # Run circuit directly
    result = run_circuit_experiment(
        circuit=circuit,
        config=config,
        simulator_seed=42,
        transpiler_seed=42
    )
    
    # Format as evaluation result
    evaluation_result = {
        'circuit': circuit,
        'results': result,
        'config': config.__dict__,
        'timestamp': result.get('timestamp', 'N/A'),
        'evaluation_type': 'direct',
        'circuit_size': circuit.num_qubits
    }
    
    return evaluation_result


def compare_evaluations(knitted_result: Dict[str, Any], direct_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare knitted and direct execution results.
    
    Args:
        knitted_result: Results from knitted execution
        direct_result: Results from direct execution
        
    Returns:
        Dictionary containing comparison metrics and analysis
    """
    comparison = {
        'knitted_results': knitted_result['results'],
        'direct_results': direct_result['results'],
        'circuit_size': knitted_result['circuit_size'],
        'config': knitted_result['config'],
        'timestamp': knitted_result['timestamp']
    }
    
    # Add basic comparison metrics
    comparison['results_match'] = knitted_result['results'] == direct_result['results']
    
    # Calculate result differences
    if comparison['results_match']:
        comparison['difference'] = 'identical'
    else:
        comparison['difference'] = 'different'
        # Could add more detailed difference analysis here
    
    return comparison


def full_evaluation(circuit: 'QuantumCircuit', start_qubit: int, end_qubit: int,
                    config: 'ExperimentConfig', num_shots: int = 1024) -> Dict[str, Any]:
    """
    Perform a complete evaluation comparing knitted vs direct execution.
    
    Args:
        circuit: Quantum circuit to evaluate
        start_qubit: Control qubit index
        end_qubit: Target qubit index
        config: Experiment configuration
        num_shots: Number of shots
        
    Returns:
        Complete comparison results
    """
    # Evaluate both approaches
    knitted = evaluate_with_knitting(circuit, start_qubit, end_qubit, config, num_shots)
    direct = evaluate_without_knitting(circuit, config, num_shots)
    
    # Compare results
    comparison = compare_evaluations(knitted, direct)
    
    # Add comprehensive metadata
    comparison['evaluation_complete'] = True
    comparison['circuit_name'] = getattr(circuit, 'name', 'unnamed_circuit')
    
    return comparison


def create_test_circuit_1() -> 'QuantumCircuit':
    """
    Create a test circuit for evaluation.
    
    Returns:
        A quantum circuit suitable for knitting evaluation
    """
    from qiskit import QuantumCircuit
    
    # Create a test circuit with multiple CNOTs for knitting
    qc = QuantumCircuit(3, 3, name='test_circuit_1')
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc


def create_test_circuit_2() -> 'QuantumCircuit':
    """
    Create another test circuit for evaluation.
    
    Returns:
        A more complex quantum circuit for evaluation
    """
    from qiskit import QuantumCircuit
    
    # Create a more complex test circuit
    qc = QuantumCircuit(4, 4, name='test_circuit_2')
    qc.h(range(4))
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 3)
    qc.cx(0, 2)
    qc.measure(range(4), range(4))
    
    return qc

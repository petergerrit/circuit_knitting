"""
Experiment execution and management functions.
"""
import pickle
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2

try:
    from .config import ExperimentConfig
    from .circuit_utils import prepare_circuit_for_execution
except ImportError:
    from config import ExperimentConfig
    from circuit_utils import prepare_circuit_for_execution


def run_circuit_experiment(
    circuit: QuantumCircuit,
    config: ExperimentConfig,
    simulator_seed: Optional[int] = None,
    transpiler_seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Run a single circuit experiment and return measurement counts.
    
    Args:
        circuit: Quantum circuit to execute
        config: Experiment configuration
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        
    Returns:
        Dictionary of measurement counts
    """
    # Set up backend based on noise configuration
    backend = FakeWashingtonV2() if config.noise else AerSimulator()
    
    # Set up transpiler
    pass_manager = generate_preset_pass_manager(
        optimization_level=config.optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed or np.random.randint(1024**2)
    )
    
    # Transpile circuit
    transpiled_circuit = pass_manager.run(circuit)
    
    # Set up sampler with options
    options = {
        "simulator": {
            "seed_simulator": simulator_seed or np.random.randint(1024**2)
        }
    }
    sampler = SamplerV2(backend, options=options)
    
    # Run experiment
    job = sampler.run([transpiled_circuit], shots=config.num_shots)
    result = job.result()[0].data.meas.get_counts()
    
    # Return sorted results
    return {key: result[key] for key in sorted(result)}


def circuit_knitter(
    circuit: QuantumCircuit,
    start_qubit: int,
    end_qubit: int,
    num_shots: int,
    config: ExperimentConfig,
    simulator_seed: Optional[int] = None,
    transpiler_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform circuit knitting on a given circuit.
    
    Args:
        circuit: Input quantum circuit
        start_qubit: Starting qubit index
        end_qubit: Ending qubit index
        num_shots: Number of shots for measurement
        config: Experiment configuration
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        
    Returns:
        Dictionary containing knitting results
    """
    # Prepare circuit for execution
    prepared_circuit = prepare_circuit_for_execution(circuit)
    
    # Update config with shot count
    updated_config = ExperimentConfig(**config.__dict__)
    updated_config.num_shots = num_shots
    
    # Run the experiment
    results = run_circuit_experiment(
        prepared_circuit,
        updated_config,
        simulator_seed=simulator_seed,
        transpiler_seed=transpiler_seed
    )
    
    return {
        'circuit': circuit,
        'results': results,
        'config': updated_config.__dict__,
        'timestamp': datetime.now().isoformat()
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
    full_path = f"{config.results_dir}/{filename}.pkl"
    
    with open(full_path, 'rb') as file:
        return pickle.load(file)
"""
Run a single circuit experiment and return measurement counts.
"""
import numpy as np
from typing import Dict, Optional
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2


def run_circuit_experiment(
    circuit: QuantumCircuit,
    config: 'ExperimentConfig',
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
    try:
        from config import ExperimentConfig
    except ImportError:
        from knitter.config import ExperimentConfig
    
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
    
    # Run job and get results
    job = sampler.run([transpiled_circuit], shots=config.num_shots)
    result_dict = job.result()[0].data.c.get_counts()
    
    return result_dict

#!/usr/bin/env python
"""
Main entry point for circuit knitting experiments.

This script demonstrates the improved organization and functionality
for running circuit knitting experiments.
"""

import numpy as np
from typing import List
from datetime import datetime

from .config import ExperimentConfig, ensure_directories
from .experiment import circuit_knitter, save_experiment_results
from .circuit_utils import trotter_stepper, create_circuit_3q_test


def run_convergence_experiment(
    step: int,
    epsilon: float,
    num_shots: int,
    noise: bool = False,
    num_seeds: int = 1
) -> List[str]:
    """
    Run convergence experiments with multiple random seeds.
    
    Args:
        step: Trotter step number
        epsilon: Epsilon value for trotter step
        num_shots: Number of shots per experiment
        noise: Whether to include noise
        num_seeds: Number of different random seeds to try
        
    Returns:
        List of paths to saved result files
    """
    # Create experiment configuration
    config = ExperimentConfig(
        noise=noise,
        num_shots=num_shots,
        epsilon_values=[epsilon],
        step_values=[step]
    )
    
    ensure_directories(config)
    
    saved_files = []
    
    for seed_idx in range(num_seeds):
        # Generate unique seed for this experiment
        experiment_seed = int(datetime.now().timestamp()) + seed_idx
        np.random.seed(experiment_seed)
        
        # Create circuit for this step
        circuit = trotter_stepper(step, config.Nqbits, epsilon, config.mass, config.mid)
        
        # Run circuit knitting experiment
        results = circuit_knitter(
            circuit=circuit,
            start_qubit=0,
            end_qubit=10,
            num_shots=num_shots,
            config=config,
            simulator_seed=np.random.randint(1024**2),
            transpiler_seed=np.random.randint(1024**2)
        )
        
        # Save results
        filename = f"step{step}_epsilon{str(epsilon).replace('.', '')}_shots{num_shots}_noise{noise}_seed{experiment_seed}"
        saved_path = save_experiment_results(results, filename, config)
        saved_files.append(saved_path)
        
        print(f"Saved results to: {saved_path}")
    
    return saved_files


def main():
    """Main function demonstrating improved circuit knitting experiments."""
    print("Circuit Knitting Improved Implementation")
    print("=" * 50)
    
    # Example: Run a convergence experiment
    print("Running example convergence experiment...")
    
    try:
        results = run_convergence_experiment(
            step=1,
            epsilon=0.2,
            num_shots=1024,
            noise=False,
            num_seeds=2
        )
        
        print(f"Successfully completed {len(results)} experiments")
        print("Results saved to:")
        for result_path in results:
            print(f"  - {result_path}")
            
    except Exception as e:
        print(f"Error running experiments: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
"""
Configuration management for circuit knitting experiments.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for circuit knitting experiments."""
    data_dir: str = "data"
    results_dir: str = "results"
    noise: bool = False
    num_shots: int = 1024
    epsilon_values: list = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    step_values: list = field(default_factory=lambda: [1, 2])
    simulator_seed: Optional[int] = None
    transpiler_seed: Optional[int] = None
    optimization_level: int = 3


def get_config() -> ExperimentConfig:
    """Get the default configuration."""
    return ExperimentConfig()


def ensure_directories(config: ExperimentConfig):
    """Ensure required directories exist."""
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
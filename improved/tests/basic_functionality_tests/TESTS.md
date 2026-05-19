The `tests/` directory contains a comprehensive test suite for the circuit knitting implementation.

- `test_simulator_randomness.py`: verifies reproducibility with fixed seeds and variability with different seeds
- `test_knitter_randomness.py`: verifies reproducibility with fixed seeds and variability with different seeds
- `test_knitter_symmetry.py`: validates symmetry between control and target qubits
- `test_simulator_noise.py`: compares noisy vs noiseless simulator execution
- `test_knitter_noise.py`: tests noise effects in the circuit knitter
- `run_all_tests.py`: orchestrates all above tests

All tests use simple circuits (Hadamard, CNOT, measurement) to isolate core functionality.
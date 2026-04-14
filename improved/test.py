# Import necessary libraries
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from circuit_utils import create_test_circuit
from experiment import circuit_knitter, ExperimentConfig

def sort_quantum_states(states_dict):
    """Sort quantum states numerically from |000...> to |111...>"""    
    # Sort by the integer value of the binary string (remove spaces first)
    return dict(sorted(states_dict.items(), key=lambda x: int(x[0].replace(' ', ''), 2) if x[0] else 0))

# Create the test circuit using the function from circuit_utils
qc = create_test_circuit()

# Add measurements to the circuit
qc.measure_all()

# Display the circuit diagram
qc.draw('mpl')

simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

# Evaluate the test circuit 16 times with (1024**2)/16 shots each
num_evaluations = 16
shots_per_evaluation = (1024**2) // 16

# Store results from each evaluation
all_results = []

# Run multiple evaluations
for i in range(num_evaluations):
    print(f'Running evaluation {i+1}/{num_evaluations}...')
    job = simulator.run(compiled_circuit, shots=shots_per_evaluation)
    result = job.result()
    counts = result.get_counts()
    all_results.append(counts)

# Combine all results
combined_counts = {}
for counts in all_results:
    for state, count in counts.items():
        if state in combined_counts:
            combined_counts[state] += count
        else:
            combined_counts[state] = count

# Calculate overall average and uncertainty
total_shots_combined = sum(combined_counts.values())
ratios = {key: value / total_shots_combined for key, value in combined_counts.items()}

# Calculate uncertainty (standard error of the mean)
import numpy as np
uncertainties = {}
for state in combined_counts.keys():
    state_counts = [counts.get(state, 0) for counts in all_results]
    std_dev = np.std(state_counts, ddof=1)
    uncertainty = std_dev / np.sqrt(num_evaluations)
    uncertainties[state] = uncertainty / total_shots_combined

# Sort quantum states numerically
ratios = sort_quantum_states(ratios)
uncertainties = sort_quantum_states(uncertainties)

# Remove trailing ' 000' from state labels for display
cleaned_states = {state.replace(' 000', ''): value for state, value in ratios.items()}
cleaned_uncertainties = {state.replace(' 000', ''): value for state, value in uncertainties.items()}

# Display results
print('\nOverall average measurement results (ratios):')
for state, ratio in cleaned_states.items():
    unc = cleaned_uncertainties.get(state, 0)
    print(f'|{state}>: {ratio:.4f} ± {unc:.6f}')

# Plot the overall average with uncertainty
states = list(cleaned_states.keys())
values = list(cleaned_states.values())
errors = list(cleaned_uncertainties.values())

plt.figure(figsize=(12, 6))
plt.bar(states, values, yerr=errors, capsize=5, alpha=0.7)
plt.title('Overall Average Measurement Results with Uncertainty')
plt.xlabel('Quantum State')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
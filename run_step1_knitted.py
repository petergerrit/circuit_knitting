#!/usr/bin/env python
"""Run step 1, no noise, knitted simulation and save results."""

import numpy as np
from circuit_utils.params import trot_step_1, mid
from circuit_utils.utils import circuit_knitter, fermion_number, bootstrap_error

# Shot values for convergence study
shots_list = [1024 * 2**i for i in range(11)]

print("Running step 1, no noise, knitted...")
np.random.seed(11)

ferm_num = []
bs_error = []

for num_shots in shots_list:
    res = circuit_knitter(trot_step_1, 0, 10, num_shots,
                          simulator_seed=np.random.randint(1024**2),
                          transpiler_seed=np.random.randint(1024**2),
                          noise=False)
    fn = fermion_number(res, mid)
    be = bootstrap_error(res, mid, num_shots)
    ferm_num.append(fn)
    bs_error.append(be)
    print(f"  Shots: {num_shots:>8}, Fermion number: {fn:.6f}, Error: {be:.6f}")

# Save to text files
with open('data/ferm_num_step1_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in ferm_num:
        f.write(f'{val}\n')

with open('data/bs_error_step1_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in bs_error:
        f.write(f'{val}\n')

print(f"\nData saved to data/ferm_num_step1_epsilon08_no_noise_knitted.txt")
print(f"Data saved to data/bs_error_step1_epsilon08_no_noise_knitted.txt")

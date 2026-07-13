#!/usr/bin/env python
"""Run step 1, no noise, no knit simulation and save results."""

import numpy as np
from circuit_utils.params import trot_step_1, mid
from circuit_utils.utils import do_run, fermion_number, bootstrap_error

# Shot values for convergence study
shots_list = [1024 * 2**i for i in range(11)]

print("Running step 1, no noise, no knit...")
np.random.seed(10)

ferm_num = []
bs_error = []

for num_shots in shots_list:
    res = do_run(trot_step_1, num_shots, noise=False,
                 simulator_seed=np.random.randint(1024**2),
                 transpiler_seed=np.random.randint(1024**2))
    fn = fermion_number(res, mid)
    be = bootstrap_error(res, mid, num_shots)
    ferm_num.append(fn)
    bs_error.append(be)
    print(f"  Shots: {num_shots:>8}, Fermion number: {fn:.6f}, Error: {be:.6f}")

# Save to text files
with open('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in ferm_num:
        f.write(f'{val}\n')

with open('data/bs_error_step1_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in bs_error:
        f.write(f'{val}\n')

print(f"\nData saved to data/ferm_num_step1_epsilon08_no_noise_no_knit.txt")
print(f"Data saved to data/bs_error_step1_epsilon08_no_noise_no_knit.txt")

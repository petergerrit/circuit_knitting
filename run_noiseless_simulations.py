#!/usr/bin/env python
# Run the four noiseless simulations: step1/step2, with/without knitting

import import_ipynb
import numpy as np

# Import from notebooks
from functions import *
from params import *

# Parameters
Nqbits = 6
Ntsteps = 3
mid = Nqbits - 2
mass = 1.125
epsilon = 0.8

# Define circuits
trot_step_1 = trotter_stepper(1, Nqbits, epsilon, mass, mid).decompose().decompose()
trot_step_2 = trotter_stepper(2, Nqbits, epsilon, mass, mid).decompose().decompose()
trot_step_1.measure_all()
trot_step_2.measure_all()

# Shot values for convergence study
shots_list = [1024 * 2**i for i in range(11)]

print("Running step 1, no noise, no knit...")
np.random.seed(10)
ferm_num_step1_no_noise_no_knit = []
bs_error_step1_no_noise_no_knit = []
for num_shots in shots_list:
    res = do_run(trot_step_1, num_shots, noise=False, 
                 simulator_seed=np.random.randint(1024**2),
                 transpiler_seed=np.random.randint(1024**2))
    ferm_num_step1_no_noise_no_knit.append(fermion_number(res, mid))
    bs_error_step1_no_noise_no_knit.append(bootstrap_error(res, mid, num_shots))
    print(f"  Shots: {num_shots}, Fermion number: {ferm_num_step1_no_noise_no_knit[-1]:.6f}")

# Save to text files
with open('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in ferm_num_step1_no_noise_no_knit:
        f.write(f'{val}\n')

with open('data/bs_error_step1_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in bs_error_step1_no_noise_no_knit:
        f.write(f'{val}\n')

print("\nRunning step 1, no noise, knitted...")
np.random.seed(11)
ferm_num_step1_no_noise_knitted = []
bs_error_step1_no_noise_knitted = []
for num_shots in shots_list:
    res = circuit_knitter(trot_step_1, 0, 10, num_shots,
                          simulator_seed=np.random.randint(1024**2),
                          transpiler_seed=np.random.randint(1024**2),
                          noise=False)
    ferm_num_step1_no_noise_knitted.append(fermion_number(res, mid))
    bs_error_step1_no_noise_knitted.append(bootstrap_error(res, mid, num_shots))
    print(f"  Shots: {num_shots}, Fermion number: {ferm_num_step1_no_noise_knitted[-1]:.6f}")

with open('data/ferm_num_step1_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in ferm_num_step1_no_noise_knitted:
        f.write(f'{val}\n')

with open('data/bs_error_step1_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in bs_error_step1_no_noise_knitted:
        f.write(f'{val}\n')

print("\nRunning step 2, no noise, no knit...")
np.random.seed(12)
ferm_num_step2_no_noise_no_knit = []
bs_error_step2_no_noise_no_knit = []
for num_shots in shots_list:
    res = do_run(trot_step_2, num_shots, noise=False,
                 simulator_seed=np.random.randint(1024**2),
                 transpiler_seed=np.random.randint(1024**2))
    ferm_num_step2_no_noise_no_knit.append(fermion_number(res, mid))
    bs_error_step2_no_noise_no_knit.append(bootstrap_error(res, mid, num_shots))
    print(f"  Shots: {num_shots}, Fermion number: {ferm_num_step2_no_noise_no_knit[-1]:.6f}")

with open('data/ferm_num_step2_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in ferm_num_step2_no_noise_no_knit:
        f.write(f'{val}\n')

with open('data/bs_error_step2_epsilon08_no_noise_no_knit.txt', 'w') as f:
    for val in bs_error_step2_no_noise_no_knit:
        f.write(f'{val}\n')

print("\nRunning step 2, no noise, knitted...")
np.random.seed(13)
ferm_num_step2_no_noise_knitted = []
bs_error_step2_no_noise_knitted = []
for num_shots in shots_list:
    res = circuit_knitter(trot_step_2, 0, 10, num_shots,
                          simulator_seed=np.random.randint(1024**2),
                          transpiler_seed=np.random.randint(1024**2),
                          noise=False)
    ferm_num_step2_no_noise_knitted.append(fermion_number(res, mid))
    bs_error_step2_no_noise_knitted.append(bootstrap_error(res, mid, num_shots))
    print(f"  Shots: {num_shots}, Fermion number: {ferm_num_step2_no_noise_knitted[-1]:.6f}")

with open('data/ferm_num_step2_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in ferm_num_step2_no_noise_knitted:
        f.write(f'{val}\n')

with open('data/bs_error_step2_epsilon08_no_noise_knitted.txt', 'w') as f:
    for val in bs_error_step2_no_noise_knitted:
        f.write(f'{val}\n')

print("\nAll simulations complete. Data saved to data/ directory.")

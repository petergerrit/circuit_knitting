#!/usr/bin/env python
"""Run step 1, no noise, knitted simulation and save results."""

import numpy as np
from circuit_utils.params import trot_step_1, mid
from circuit_utils.utils import circuit_knitter, fermion_number, bootstrap_error

# Shot values for convergence study
shots_list = [1024 * 2**i for i in range(11)]

np.random.seed(11)

# Open output files for incremental writing
with open('data/ferm_num_step1_epsilon08_no_noise_knitted.txt', 'w') as f_fn, \
     open('data/bs_error_step1_epsilon08_no_noise_knitted.txt', 'w') as f_be:
    for num_shots in shots_list:
        res = circuit_knitter(trot_step_1, 0, 10, num_shots,
                              simulator_seed=np.random.randint(1024**2),
                              transpiler_seed=np.random.randint(1024**2),
                              noise=False)
        fn = fermion_number(res, mid)
        be = bootstrap_error(res, mid, num_shots)
        f_fn.write(f'{fn}\n')
        f_be.write(f'{be}\n')

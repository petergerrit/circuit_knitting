#!/usr/bin/env python
"""Run step 1, no noise, no knit simulation and save results."""

import numpy as np
from circuit_utils.params import trot_step_1, mid
from circuit_utils.utils import do_run, fermion_number, bootstrap_error

# Shot values for convergence study
shots_list = [1024 * 2**i for i in range(11)]

np.random.seed(10)

# Open output files for incremental writing
with open('data/ferm_num_step1_epsilon08_no_noise_no_knit.txt', 'w') as f_fn, \
     open('data/bs_error_step1_epsilon08_no_noise_no_knit.txt', 'w') as f_be:
    for num_shots in shots_list:
        res = do_run(trot_step_1, num_shots, noise=False,
                     simulator_seed=np.random.randint(1024**2),
                     transpiler_seed=np.random.randint(1024**2))
        fn = fermion_number(res, mid)
        be = bootstrap_error(res, mid, num_shots)
        f_fn.write(f'{fn}\n')
        f_fn.flush()
        f_be.write(f'{be}\n')
        f_be.flush()

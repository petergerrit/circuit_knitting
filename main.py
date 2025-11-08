#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_aer.primitives import EstimatorV2

from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

from functions import *
from params import *


# In[2]:


# Nqbits = 6
# Ntsteps = 3
# mid = Nqbits - 2
# mass = 1.125
# epsilon = 0.8


# In[4]:


# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_no_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_1, 0, 10, num_shots), file, protocol=pickle.HIGHEST_PROTOCOL)


# In[5]:


# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_1, 0, 10, num_shots, noise=True), file, protocol=pickle.HIGHEST_PROTOCOL)


# In[6]:


# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_no_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_2, 0, 10, num_shots), file, protocol=pickle.HIGHEST_PROTOCOL)


# In[7]:


for i in range(11):
    num_shots = 1024*2**i
    with open('data/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
        pickle.dump(circuit_knitter(trot_step_2, 0, 10, num_shots, noise=True), file, protocol=pickle.HIGHEST_PROTOCOL)


# In[8]:


# with open('data/step1_epsilon08_count16_no_noise.pkl', 'rb') as file:
#     temp_res = pickle.load(file)


# In[9]:


# fermion_number(temp_res, mid)


# In[10]:


# bootstrap_error(temp_res, mid, 16)


# In[ ]:





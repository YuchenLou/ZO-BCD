#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Simple example of using ZO-BCD

TODO: Test ZO-BCD-RC

"""

from optimizers import *
from benchmarkfunctions import *

import numpy as np
import matplotlib.pyplot as plt


######################### Problem set up #####################################
n = 20000 # problem dimension
s_exact = 200   # True sparsity 
noiseamp = 0.001  # noise amplitude 
obj_func = MaxK(n, s_exact, noiseamp)  # initialize objective function


######################## Initialize ZO-BCD ###################################


x0    = np.random.randn(n) # initial iterate
step_size = 0.5
params = {"Type": "ZOBCD-R", "sparsity": s_exact, "delta": 0.001, "J": 20}
function_budget = 1e4
## the following are optional settings
# shuffle = False
# function_target = 0.01

performance_log = [] # initialize logging

opt = ZOBCD(x0, step_size, obj_func, params, function_budget)


####################### Iterate Optimizer ###################################

termination = False
prev_evals = 0

while termination is False:
    # optimization step
    curr_evals, solution, termination = opt.step()
    f_est = opt.f_est # function value BEFORE taking current step.
    performance_log.append([prev_evals, f_est])
    prev_evals = curr_evals
    
    # if desired, print some useful information
    print('Function value: %f  Num. evals %d\n' % (f_est, prev_evals))

    
####################### Plot the results ####################################


plt.plot(np.array(performance_log)[:,0],
         np.log10(np.array(performance_log)[:,1]), linewidth=1)
plt.title('Log-y plot of function value vs. number of queries')
plt.show()

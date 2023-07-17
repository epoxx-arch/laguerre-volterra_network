#!python3

# Copyright (C) 2023  Victor O. Costa

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Own
from laguerre_volterra_network import LVN
import optimization_utilities as ou
from data_handling import read_io
## MH
import ant_colony_for_continuous_domains
import simulated_annealing
import particle_swarm_optimization
# Pyython std library
# import math
# import sys
# 3rd party
import numpy as np

#
train_filename = './signals_and_systems/infinite_order_train.csv'
test_filename = './signals_and_systems/infinite_order_test.csv'
train_in, train_out = read_io(train_filename)
test_in, test_out = read_io(test_filename)


## Structure
L = 3; H = 3; Q = 5
bo_link = True
Fs = 25

print(f'(L,H,Q) = ({L},{H},{Q})')

## [1] Define fitness used to optimize ALPHA and W
## Polynomial coefficients are computed as a LSTSQ solution
encoding_scheme = 1
cost_alpha_weights = ou.define_cost(encoding_scheme, L, H, Q, bo_link, Fs, train_filename)

# Optimization parameters
function_evals = [100, 150, 200]
ntimes = 30
solution_encoding = 1
m = 5; k = 50; q = 0.01; xi = 0.85

# Setup metaheuristics
## Parameters to be optimized
alpha_min   = 1e-5;     alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263

# 
weights_ranges = [1e-5, 1, 1e5]
acor_optimizers = []

# Define one optimizer for each item in weights_ranges
for wrange in weights_ranges:
    weight_max = wrange
    weight_min = wrange * -1
    
    optimization_ranges = []
    optimization_bounding = []
    
    # Alpha
    optimization_ranges.append([alpha_min, alpha_max])
    optimization_bounding.append(True)
    
    # Hidden units weights
    for _ in range(L*H):
        optimization_ranges.append([weight_min, weight_max])
        optimization_bounding.append(False)
    
    # 
    train_costs = []
    test_costs = []
    
    ACOr_aw = ant_colony_for_continuous_domains.ACOr()
    ACOr_aw.set_verbosity(False)
    ACOr_aw.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
    ACOr_aw.set_parameters(m, k, q, xi, function_evals)
    ACOr_aw.define_variables(optimization_ranges, optimization_bounding)

    for _ in range(ntimes):
        # Train
        aw_solutions = ACOr_aw.optimize()
        aw_cost_history = (np.array(aw_solutions))[:, -1]
        train_costs.append(aw_cost_history)
        
        # 
        aw_cost_history_test = []
        
        # Test
        for aw in aw_solutions:
            # 
            model = LVN(L, H, Q, 1 / Fs, bo_link)
            alpha, W = ou.decode_alpha_weights(aw, L, H)
            model.set_connection_weights(W)
            C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
            model.set_polynomial_coefficients(C)
            pred_test_out = model.predict(test_in, alpha)
            cost = ou.NMSE(test_out, pred_test_out, alpha)
            aw_cost_history_test.append(cost)
            
        
        # 
        test_costs.append(aw_cost_history_test)
        
    # Train cost history
    avg_train_history = np.sum(train_costs, axis=0) / ntimes
    
    # Test cost history
    avg_test_history = np.sum(test_costs, axis=0) / ntimes
    
    print(f'WRanges = [{wrange}, {wrange * -1}]')
    print(f'[TRAIN] {np.shape(avg_train_history)} {avg_train_history}')
    print(f'[TEST] {np.shape(avg_test_history)} {avg_test_history}')


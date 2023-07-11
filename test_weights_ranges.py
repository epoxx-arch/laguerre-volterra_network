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

# Setup metaheuristics
## Parameters to be optimized
alpha_min   = 1e-5;     alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263
weight_min0  = -1e-5;   weight_max0  = 1e-5
weight_min1  = -1;      weight_max1  = 1
weight_min2  = -1e5;    weight_max2  = 1e5

# Define the ranges to be used in random initialization of algorithms for each variable,
#  along with which variables are bounded by these ranges during the optimization
aw_ranges0 = []
aw_ranges1 = []
aw_ranges2 = []
aw_bounding = []

# Alpha variable is bounded
aw_ranges0.append([alpha_min, alpha_max])
aw_ranges1.append([alpha_min, alpha_max])
aw_ranges2.append([alpha_min, alpha_max])
aw_bounding.append(True)

# Hidden units input weights are forcedly bounded by l2-normalization (normalization to unit Euclidean norm)
for _ in range(L * H): 
    aw_ranges0.append([weight_min0, weight_max0])
    aw_ranges1.append([weight_min1, weight_max1])
    aw_ranges2.append([weight_min2, weight_max2])
    aw_bounding.append(False)

# Optimization
function_evals = [100, 150, 200, 250, 300]
ntimes = 30
#
aw0_train_costs = []
aw1_train_costs = []
aw2_train_costs = []
aw0_test_costs = []
aw1_test_costs = []
aw2_test_costs = []

# ACOr shared params
solution_encoding = 1
m = 5; k = 50; q = 0.01; xi = 0.85

# Ant Colony Optimization [wrange: +- 1e-5]
ACOr_aw0 = ant_colony_for_continuous_domains.ACOr()
ACOr_aw0.set_verbosity(False)
ACOr_aw0.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_aw0.set_parameters(m, k, q, xi, function_evals)
ACOr_aw0.define_variables(aw_ranges0, aw_bounding)

# Ant Colony Optimization [wrange: +- 1]
ACOr_aw1 = ant_colony_for_continuous_domains.ACOr()
ACOr_aw1.set_verbosity(False)
ACOr_aw1.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_aw1.set_parameters(m, k, q, xi, function_evals)
ACOr_aw1.define_variables(aw_ranges1, aw_bounding)

# Ant Colony Optimization [wrange: +- 1e5]
ACOr_aw2 = ant_colony_for_continuous_domains.ACOr()
ACOr_aw2.set_verbosity(False)
ACOr_aw2.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_aw2.set_parameters(m, k, q, xi, function_evals)
ACOr_aw2.define_variables(aw_ranges2, aw_bounding)

#
for _ in range(ntimes):
    # Train
    ## [wrange: +- 1e-5]
    aw0_solutions = ACOr_aw0.optimize()
    aw0_cost_history = (np.array(aw0_solutions))[:, -1]
    aw0_train_costs.append(aw0_cost_history)
    ## [wrange: +- 1]
    aw1_solutions = ACOr_aw1.optimize()
    aw1_cost_history = (np.array(aw1_solutions))[:, -1]
    aw1_train_costs.append(aw1_cost_history)
    ## [wrange: +- 1e5]
    aw2_solutions = ACOr_aw2.optimize()
    aw2_cost_history = (np.array(aw2_solutions))[:, -1]
    aw2_train_costs.append(aw2_cost_history)
    # 
    aw0_cost_history_test = []
    aw1_cost_history_test = []
    aw2_cost_history_test = []
    
    # Test
    for aw0, aw1, aw2 in zip(aw0_solutions, aw1_solutions, aw2_solutions):
        # AW0
        model = LVN(L, H, Q, 1 / Fs, bo_link)
        alpha, W = ou.decode_alpha_weights(aw0, L, H)
        model.set_connection_weights(W)
        C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
        model.set_polynomial_coefficients(C)
        pred_test_out = model.predict(test_in, alpha)
        cost = ou.NMSE(test_out, pred_test_out, alpha)
        aw0_cost_history_test.append(cost)
        
        # AW1
        model = LVN(L, H, Q, 1 / Fs, bo_link)
        alpha, W = ou.decode_alpha_weights(aw1, L, H)
        model.set_connection_weights(W)
        C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
        model.set_polynomial_coefficients(C)
        pred_test_out = model.predict(test_in, alpha)
        cost = ou.NMSE(test_out, pred_test_out, alpha)
        aw1_cost_history_test.append(cost)
        
        # AW2
        model = LVN(L, H, Q, 1 / Fs, bo_link)
        alpha, W = ou.decode_alpha_weights(aw2, L, H)
        model.set_connection_weights(W)
        C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
        model.set_polynomial_coefficients(C)
        pred_test_out = model.predict(test_in, alpha)
        cost = ou.NMSE(test_out, pred_test_out, alpha)
        aw2_cost_history_test.append(cost)
    
    # 
    aw0_test_costs.append(aw0_cost_history_test)
    aw1_test_costs.append(aw1_cost_history_test)
    aw2_test_costs.append(aw2_cost_history_test)
    

# Train cost history
aw0_avg_train_history = np.sum(aw0_train_costs, axis=0) / ntimes
aw1_avg_train_history = np.sum(aw1_train_costs, axis=0) / ntimes
aw2_avg_train_history = np.sum(aw2_train_costs, axis=0) / ntimes

# Test cost history
aw0_avg_test_history = np.sum(aw0_test_costs, axis=0) / ntimes
aw1_avg_test_history = np.sum(aw1_test_costs, axis=0) / ntimes
aw2_avg_test_history = np.sum(aw2_test_costs, axis=0) / ntimes

print('[TRAIN]')
print(f'ACOr AW0 {np.shape(aw0_avg_train_history)} {aw0_avg_train_history}')
print(f'ACOr AW1 {np.shape(aw1_avg_train_history)} {aw1_avg_train_history}')
print(f'ACOr AW2 {np.shape(aw2_avg_train_history)} {aw2_avg_train_history}')

print('[TEST]')
print(f'ACOr AW0 {np.shape(aw0_avg_test_history)} {aw0_avg_test_history}')
print(f'ACOr AW1 {np.shape(aw1_avg_test_history)} {aw1_avg_test_history}')
print(f'ACOr AW2 {np.shape(aw2_avg_test_history)} {aw2_avg_test_history}')

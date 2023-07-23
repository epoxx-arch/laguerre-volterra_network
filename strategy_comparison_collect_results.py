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
# Pyython std library
import math
import sys
# 3rd party
import numpy as np

# Argument number checking
if len(sys.argv) != 3:
    print('Error, wrong number of arguments. Execute this script as follows:\npython3 %s {dataset order} {optimization strategy}' % sys.argv[0])
    exit(-1)

# Argument coherence checking
order_str = sys.argv[1] 
if order_str != 'finite' and order_str != 'infinite':
    print('Error, choose either \'finite\' or \'infinite\' for the simulated system order')
    exit(-1)

optimization_strategy = sys.argv[2] 
if optimization_strategy != '0' and optimization_strategy != '1' and optimization_strategy != '2':
    print('Error, choose either \'0\' or \'1\' or \'2\' for the simulated system order')
    exit(-1)

# Optimization strategies:
# [0] Define fitness used to optimize ALPHA and RANGE of random weights
## Weights are randomized and polynomial coefficients are computed as a least-squares solution
# [1] Define fitness used to optimize ALPHA and W
## Polynomial coefficients are computed as a least-squares solution
# [2] Define fitness used to optimize ALPHA, W and C
optimization_strategy = int(optimization_strategy)

# Optimized LVN structure definition
Fs = 25                             # Sampling frequency is assumed to be 25 Hz, but could be any other value
L = None;   H = None;    Q = None;    

# Load data from parameterizable file name
train_filename = './signals_and_systems/' + order_str + '_order_train.csv'
test_filename  = './signals_and_systems/' + order_str + '_order_test.csv'

train_in, train_out = read_io(train_filename)
test_in, test_out = read_io(test_filename)

# Define LVN structure
bo_link = False
Fs = 25

if order_str == 'finite':
    L = 5;  H = 3;  Q = 4
else:
    L = 2;  H = 3;  Q = 5

print(f'(L,H,Q) = ({L},{H},{Q})')

# Setup metaheuristics
## Parameters to be optimized
alpha_min   = 1e-5; alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263
wrange_min  = 0.5;  wrange_max  = 128   # Based on previous weights ranges analysis (weights_ranges_tests.py) 
weight_min  = -1;   weight_max  = 1
coef_min    = -1;   coef_max    = 1

# Define the ranges to be used in random initialization of algorithms for each variable,
#  along with which variables are bounded by these ranges during the optimization
variables_ranges = []
variables_bounding = []

# Alpha variable is bounded
variables_ranges.append([alpha_min, alpha_max])
variables_bounding.append(True)

# Weight ranges are bounded to initial ranges
if optimization_strategy == 0:
    variables_ranges.append([wrange_min, wrange_max])
    variables_bounding.append(True)

# Hidden units input weights are not bounded to initial ranges
if optimization_strategy == 1 or optimization_strategy == 2:
    for _ in range(L * H): 
        variables_ranges.append([weight_min, weight_max])
        variables_bounding.append(True)

# Polynomial coefficients are not bounded in the initial range
if optimization_strategy == 2:
    # The presence of a bank-output link defines the number of poly coefficients
    if bo_link:
        num_coefs = H * (Q - 1) + L + 1
    else:
        num_coefs = H * Q + 1
    
    # 
    for _ in range(num_coefs):
        variables_ranges.append([coef_min, coef_max])
        variables_bounding.append(True)

# Optimization
function_evals = np.arange(100, 10100, 100)
ntimes = 30

# Define cost functions according to the optimization strategy
cost_function = ou.define_cost(optimization_strategy, L, H, Q, bo_link, Fs, train_filename)

# Ant Colony Optimization
print(f'ACOr with optimization strategy [{optimization_strategy}]')
m = 5; k = 50; q = 0.01; xi = 0.85
metaheuristic = ant_colony_for_continuous_domains.ACOr()
metaheuristic.set_verbosity(False)
metaheuristic.set_cost(cost_function)
metaheuristic.set_parameters(m, k, q, xi, function_evals)
metaheuristic.define_variables(variables_ranges, variables_bounding)

#
train_solutions = []
train_costs = []
test_costs = []

for _ in range(ntimes):
    #
    solutions_history = metaheuristic.optimize()
    cost_history_train = (np.array(solutions_history))[:, -1]
    #
    train_solutions.append(solutions_history)
    train_costs.append(cost_history_train)
    
    # 
    cost_history_test = []
    for solution in solutions_history:
        # 
        model = LVN(L, H, Q, 1 / Fs, bo_link)
        
        # AR strategy
        if optimization_strategy == 0:
            # print('AR')
            model = LVN(L, H, Q, 1 / Fs, bo_link)
            alpha, range = ou.decode_alpha_range(solution)
            # print(alpha, range)
            W = ou.randomize_weights(range, L, H)
            model.set_connection_weights(W)
            C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
            model.set_polynomial_coefficients(C)
            #
            pred_test_out = model.predict(test_in, alpha)
            test_cost = ou.NMSE(test_out, pred_test_out, alpha)
            cost_history_test.append(test_cost)
        
        # AW strategy
        elif optimization_strategy == 1:
            # print('AW')
            alpha, W = ou.decode_alpha_weights(solution, L, H)
            # print(alpha, W)
            model.set_connection_weights(W)
            C = ou.train_poly_least_squares(model, train_in, train_out, alpha)
            model.set_polynomial_coefficients(C)
            #
            pred_test_out = model.predict(test_in, alpha)
            test_cost = ou.NMSE(test_out, pred_test_out, alpha)
            cost_history_test.append(test_cost)
        
        # AWC strategy
        else:
            # print('AWC')
            alpha, W, C = ou.decode_alpha_weights_coefficients(solution, L, H, Q, bo_link)
            # print(alpha, W, C)        
            model.set_connection_weights(W)
            model.set_polynomial_coefficients(C)
            #
            pred_test_out = model.predict(test_in, alpha)
            test_cost = ou.NMSE(test_out, pred_test_out, alpha)
            cost_history_test.append(test_cost)
    
    # 
    test_costs.append(cost_history_test)
    
# Compute train and test cost history
avg_train_history = np.sum(train_costs, axis=0) / ntimes
avg_test_history = np.sum(test_costs, axis=0) / ntimes

# 
print(f'Data shapes: {np.shape(train_solutions)} {np.shape(avg_train_history)} {np.shape(avg_test_history)}')
print(f'[TRAIN] Avg cost history: {avg_train_history}')
print(f'[TEST]  Avg cost history: {avg_test_history}')

#
output_base_filename = str(optimization_strategy) 
np.save('./data/strategy_' + str(optimization_strategy) + '_' + order_str + '_train_solutions.npy', train_solutions)
np.save('./data/strategy_' + str(optimization_strategy) + '_' + order_str + '_train_costs.npy', train_costs)
np.save('./data/strategy_' + str(optimization_strategy) + '_' + order_str + '_test_costs.npy' , test_costs)

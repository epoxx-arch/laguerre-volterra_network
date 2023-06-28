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
from reservoir_laguerre_volterra_network_structure import RLVN
import ant_colony_for_continuous_domains
import simulated_annealing
import particle_swarm_optimization
import optimization_utilities as ou
from data_handling import read_io
# Pyython std library
import math
import sys
# 3rd party
import numpy as np

#
train_filename = './signals_and_systems/short_finite_train.csv'
train_in, train_out = read_io(train_filename)

## Structure
L = 3; H = 5; Q = 2
bo_link = False
Fs = 25

print(f'(L,H,Q) = ({L},{H},{Q})')

# Define and evaluate models for all three solution coding schemes of the cost functions

## [0] Define fitness used to optimize ALPHA and RANGE of random weights
## Weights are randomized and polynomial coefficients are computed as a LSTSQ solution
encoding_scheme = 0
cost_alpha_range = ou.define_cost(encoding_scheme, L, H, Q, bo_link, Fs, train_filename)

## [1] Define fitness used to optimize ALPHA and W
## Polynomial coefficients are computed as a LSTSQ solution
encoding_scheme = 1
cost_alpha_weights = ou.define_cost(encoding_scheme, L, H, Q, bo_link, Fs, train_filename)

## [2] Define fitness used to optimize ALPHA, W and C
encoding_scheme = 2
cost_alpha_weights_coef = ou.define_cost(encoding_scheme, L, H, Q, bo_link, Fs, train_filename)

# Setup metaheuristics
## Parameters to be optimized
alpha_min   = 1e-5; alpha_max   = 0.9   # estimated lag with alpha = 0.9 is 263
weight_min  = -1;   weight_max  = 1
wrange_min  = -5;   wrange_max  = 5
coef_min    = -1;   coef_max    = 1  

# Define the ranges to be used in random initialization of algorithms for each variable,
#  along with which variables are bounded by these ranges during the optimization
awc_ranges = []
aw_ranges = []
ar_ranges = []
awc_bounding = []
aw_bounding = []
ar_bounding = []

# Alpha variable is bounded
awc_ranges.append([alpha_min, alpha_max])
awc_bounding.append(True)
aw_ranges.append([alpha_min, alpha_max])
aw_bounding.append(True)
ar_ranges.append([alpha_min, alpha_max])
ar_bounding.append(True)

# Hidden units input weights are forcedly bounded by l2-normalization (normalization to unit Euclidean norm)
for _ in range(L * H): 
    awc_ranges.append([weight_min, weight_max])
    awc_bounding.append(False)
    aw_ranges.append([weight_min, weight_max])
    aw_bounding.append(False)

# Polynomial coefficients are not bounded in the initial range
if bo_link:
    num_coefs = H * (Q - 1) + L + 1
else:
    num_coefs = H * Q + 1

for _ in range(num_coefs):
    awc_ranges.append([coef_min, coef_max])
    awc_bounding.append(False)

# Weight ranges optimization
ar_ranges.append([wrange_min, wrange_max])
ar_bounding.append(False)

# Optimization
function_evals = [75, 100, 125, 150, 175, 200]
ntimes = 30

#
awc_costs = []
aw_costs = []
ar_costs = []

# ACOr optimizing encoding scheme 0: 
# Total # of function evaluations: archive_size + population_size * num_iterations
solution_encoding = 0
print(f'ACOr AR [{solution_encoding}]')
m = 5; k = 50; q = 0.01; xi = 0.85
ACOr_ar = ant_colony_for_continuous_domains.ACOr()
ACOr_ar.set_verbosity(False)
ACOr_ar.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_ar.set_parameters(m, k, q, xi, function_evals)
ACOr_ar.define_variables(ar_ranges, ar_bounding)

# ACOr optimizing encoding scheme 1: 
# Total # of function evaluations: archive_size + population_size * num_iterations
solution_encoding = 1
print(f'ACOr AW [{solution_encoding}]')
m = 5; k = 50; q = 0.01; xi = 0.85
ACOr_aw = ant_colony_for_continuous_domains.ACOr()
ACOr_aw.set_verbosity(False)
ACOr_aw.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_aw.set_parameters(m, k, q, xi, function_evals)
ACOr_aw.define_variables(aw_ranges, aw_bounding)

# ACOr optimizing encoding scheme 2: 
# Total # of function evaluations: archive_size + population_size * num_iterations
solution_encoding = 2
print(f'ACOr AWC [{solution_encoding}]')
m = 5; k = 50; q = 0.01; xi = 0.85
ACOr_awc = ant_colony_for_continuous_domains.ACOr()
ACOr_awc.set_verbosity(False)
ACOr_awc.set_cost(ou.define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename))
ACOr_awc.set_parameters(m, k, q, xi, function_evals)
ACOr_awc.define_variables(awc_ranges, awc_bounding)

#
for _ in range(ntimes):
    #
    best_solution = ACOr_ar.optimize()
    best_cost = (np.array(best_solution))[:, -1]
    ar_costs.append(best_cost)
    #
    best_solution = ACOr_aw.optimize()
    best_cost = (np.array(best_solution))[:, -1]
    aw_costs.append(best_cost)
    #
    best_solution = ACOr_awc.optimize()
    best_cost = (best_solution)[:, -1]
    awc_costs.append(np.array(best_cost))
    
# print(f'ACOr AR {np.shape(ar_costs)} {ar_costs})')
# print(f'ACOr AW {np.shape(aw_costs)} {aw_costs}')
# print(f'ACOr AWC {np.shape(awc_costs)} {awc_costs}')

ar_avg_history = np.sum(ar_costs, axis=0) / ntimes
aw_avg_history = np.sum(aw_costs, axis=0) / ntimes
awc_avg_history = np.sum(awc_costs, axis=0) / ntimes

print(f'ACOr AR {np.shape(ar_avg_history)} {ar_avg_history})')
print(f'ACOr AW {np.shape(aw_avg_history)} {aw_avg_history}')
print(f'ACOr AWC {np.shape(awc_avg_history)} {awc_avg_history}')
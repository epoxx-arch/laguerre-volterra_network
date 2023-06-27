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
from data_handling import read_io
from reservoir_laguerre_volterra_network_structure import RLVN
import optimization_utilities
# Pyython std library
import math
import sys
# 3rd party
import numpy as np

if len(sys.argv) != 3:
    print('Error, use as ' + sys.argv[0] + ' \'short\'|\'long\ \'finite\'|\'infinite\'')
    exit(-1)
    
data_size = (sys.argv[1]).lower()
data_order = (sys.argv[2]).lower()
if data_size != 'short' and data_size != 'long':
    print('Error, size must be \'short\' or \'long\'')
    exit(-1)
if data_order != 'finite' and data_order != 'infinite':
    print('Erro r, order must be \'finite\' or \'infinite\'')
    exit(-1)    
    
print(f'Size = {data_size}, order = {data_order}')
    
train_in, train_out = read_io('./signals_and_systems/' + data_size + '_' + data_order + '_train.csv')
test_in, test_out   = read_io('./signals_and_systems/' + data_size + '_' + data_order + '_test.csv')

## Configs
# Structure
L = 3; H = 10; Q = 4
Fs = 25
# Abs range of random weights
wrange = 0.1
# Laguerre smoothing constant
alpha = 0.7
# Number of runs
ntimes = 10
#
l2_regularization = True
if l2_regularization:
    print('\nUSING L2 REGULARIZATION\n')

# Randomize weights a single time to reduce variability in comparisons
random_weights = optimization_utilities.randomize_weights(wrange, L, H)
print(f'(L,H,Q) = ({L},{H},{Q})')

# RLVN with H maps from filter bank to hidden layer (same as original LVN)

# Without bank-output link
bo_link = False
bo_false_train_errors = []
bo_false_test_errors = []

for _ in range(ntimes):
    # Define model parameters and set connection weights as random numbers
    model = RLVN(L, H, Q, 1 / Fs, bo_link)
    model.set_connection_weights(random_weights)
    
    #
    poly_coefficients = optimization_utilities.train_poly_least_squares(model,
                          train_in, train_out, alpha)                  
    model.set_polynomial_coefficients(poly_coefficients)
    
    #
    estimated_train_out = model.predict(train_in, alpha)
    estimated_test_out = model.predict(test_in, alpha)
    
    # Compute and keep errors
    nmse_train = optimization_utilities.NMSE(train_out, estimated_train_out, alpha)
    nmse_test = optimization_utilities.NMSE(test_out, estimated_test_out, alpha)
    bo_false_train_errors.append(nmse_train)
    bo_false_test_errors.append(nmse_test)

# With bank-output link
bo_link = True
bo_true_train_errors = []
bo_true_test_errors = []

for _ in range(ntimes):
    # Define model parameters and set connection weights as random numbers
    model = RLVN(L, H, Q, 1 / Fs, bo_link)
    model.set_connection_weights(random_weights)
    
    #
    poly_coefficients = optimization_utilities.train_poly_least_squares(model,
                          train_in, train_out, alpha)
    model.set_polynomial_coefficients(poly_coefficients)
    
    #
    estimated_train_out = model.predict(train_in, alpha)
    estimated_test_out = model.predict(test_in, alpha)
    
    # Compute and keep errors
    nmse_train = optimization_utilities.NMSE(train_out, estimated_train_out, alpha)
    nmse_test = optimization_utilities.NMSE(test_out, estimated_test_out, alpha)
    bo_true_train_errors.append(nmse_train)
    bo_true_test_errors.append(nmse_test)


print('Train')
print(f'NMSE without bo: {np.mean(bo_false_train_errors)} ({np.std(bo_false_train_errors, ddof = -1)})')
print(f'NMSE with bo: {np.mean(bo_true_train_errors)} ({np.std(bo_true_train_errors, ddof = -1)})')
print('Test')
print(f'NMSE without bo: {np.mean(bo_false_test_errors)} ({np.std(bo_false_test_errors, ddof = -1)})')
print(f'NMSE with bo: {np.mean(bo_true_test_errors)} ({np.std(bo_true_test_errors, ddof = -1)})')
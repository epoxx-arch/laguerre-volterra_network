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
from laguerre_volterra_network import LVN
import optimization_utilities as ou
# Pyython std library
import math
import sys
# 3rd party
import numpy as np

# PADRONIZAR NOMES DE ARQUIVOS
train_filename = './signals_and_systems/short_finite_train.csv'
train_in, train_out = read_io(train_filename)

# print(f'Train len: {len(train_in)}  Test len: {len(test_in)}')

# print(train_in)
        
## Structure
L = 3; H = 2; Q = 2
print(f'(L,H,Q) = ({L},{H},{Q})')
Fs = 25


# Define and evaluate models for all three solution coding schemes of the cost functions
#   For solution_encoding = 2: Optimize alphas, weights and coefficients
#   For solution_encoding = 1: Optimize alphas and weights
#   For solution_encoding = 0: Optimize alphas and weights range

# Define model given ALPHA and RANGE of random weights
## Parameters
alpha = 0.7
wrange = 0.1
bo_link = False

#
solution = [alpha, wrange]
alpha, wrange = ou.decode_alpha_range(solution)
print(f'TEST: alpha is {alpha} and wrange is {wrange}')

model = LVN(L, H, Q, 1 / Fs, bo_link)
random_weights = ou.randomize_weights(wrange, L, H)
model.set_connection_weights(random_weights)
#
poly_coefficients = ou.train_poly_least_squares(model,
                      train_in, train_out, alpha)
model.set_polynomial_coefficients(poly_coefficients)
#
model_train_out = model.predict(train_in, alpha)
nmse_train = ou.NMSE(train_out, model_train_out, alpha)

print(f'Train NMSE [0] : {nmse_train}')
# print(f'Test NMSE [0]: {nmse_train}')


# Define model given ALPHA and WEIGHTS
## Parameters
alpha = 0.7
# W = [[1, 1.5],[2, 2.5],[3, 3.5]]
W = [1, 1.5, 2, 2.5, 3, 3.5]
bo_link = False

solution = [alpha] + W
alpha, W = ou.decode_alpha_weights(solution, L, H)
print(f'TEST: alpha is {alpha} and W is {W}')

model = LVN(L, H, Q, 1 / Fs, bo_link)
model.set_connection_weights(W)
#
poly_coefficients = ou.train_poly_least_squares(model,
                      train_in, train_out, alpha)
model.set_polynomial_coefficients(poly_coefficients)
#
model_train_out = model.predict(train_in, alpha)
nmse_train = ou.NMSE(train_out, model_train_out, alpha)

print(f'Train NMSE [1] : {nmse_train}')

# Define model given ALPHA, WEIGHTS and COEFFICIENTS
## Parameters
alpha = 0.7
# W = [[1, 1.5],[2, 2.5],[3, 3.5]]
W = [1, 1.5, 2, 2.5, 3, 3.5]
C = [0.5, 2, 3, 4, 5]
bo_link = False

solution = [alpha] + W + C
alpha, W, C = ou.decode_alpha_weights_coefficients(solution, L, H, Q, bo_link)
print(f'TEST: alpha is {alpha}, W is {W} and C is {C}')

model = LVN(L, H, Q, 1 / Fs, bo_link)
model.set_connection_weights(W)
model.set_polynomial_coefficients(C)
#
model_train_out = model.predict(train_in, alpha)
nmse_train = ou.NMSE(train_out, model_train_out, alpha)

print(f'Train NMSE [2.1] : {nmse_train}')

# With BO link
alpha = 0.7
# W = [[1, 1.5],[2, 2.5],[3, 3.5]]
W = [1, 1.5, 2, 2.5, 3, 3.5]
C = [0.5, 2, 3, 4, 5, 6]
bo_link = True

solution = [alpha] + W + C
print(solution)
alpha, W, C = ou.decode_alpha_weights_coefficients(solution, L, H, Q, bo_link)
print(f'TEST: alpha is {alpha}, W is {W} and C is {C}')

model = LVN(L, H, Q, 1 / Fs, bo_link)
model.set_connection_weights(W)
model.set_polynomial_coefficients(C)
#
model_train_out = model.predict(train_in, alpha)
nmse_train = ou.NMSE(train_out, model_train_out, alpha)

print(f'Train NMSE [2.2] : {nmse_train}')

#
# Define cost functions

bo_link = False
cost_0 = ou.define_cost(0, L, H, Q, bo_link, Fs, train_filename)
cost_1 = ou.define_cost(1, L, H, Q, bo_link, Fs, train_filename)
cost_2_1 = ou.define_cost(2, L, H, Q, bo_link, Fs, train_filename)

bo_link = True
cost_2_2 = ou.define_cost(2, L, H, Q, bo_link, Fs, train_filename)

# Compute cost given ALPHA and RANGE of random weights
## Parameters
alpha = 0.7
wrange = 0.1

solution = [alpha, wrange]
nmse = cost_0(solution)
print(f'CF NMSE [0] : {nmse}')

# Compute cost given ALPHA and W
## Parameters
W = [1, 1.5, 2, 2.5, 3, 3.5]

solution = [alpha] + W
print(solution)
nmse = cost_1(solution)
print(f'CF NMSE [1] : {nmse}')

# Compute cost given ALPHA, W and C
## Parameters
# Without BO link
C = [0.5, 2, 3, 4, 5]

solution = [alpha] + W + C
print(solution)
nmse = cost_2_1(solution)
print(f'CF NMSE [2.1] : {nmse}')

# With BO link 
C = [0.5, 2, 3, 4, 5, 6]

solution = [alpha] + W + C
print(solution)
nmse = cost_2_2(solution)
print(f'CF NMSE [2.2] : {nmse}')
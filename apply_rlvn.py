#!python3

# Copyright (C) 2020  Victor O. Costa

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
from optimization_utilities import NMSE
from reservoir_laguerre_volterra_network_structure import RLVN
# Pyython std library
import math
# 3rd party
import numpy as np

L = 5
H = 3
Q = 4
Fs = 25
model = RLVN(L,H,Q, Fs)

alpha = 0.2348794219586014
weights_range = 1
sig_in, sig_out = read_io('./signals_and_systems/finite_order_train.csv')
test_in, test_out = read_io('./signals_and_systems/finite_order_test.csv')

weights, fm = model.compute_feature_matrix_rand(sig_in, alpha, weights_range)
# print(fm)
# print(np.linalg.matrix_rank(fm))

print('[least squares solution]')
beta, _, rank, _ = np.linalg.lstsq(fm, sig_out)
print(np.shape(beta))
print(beta)

# print('out')
estimated_train_out  = np.matmul(fm, beta)
# print(np.shape(estimated_train_out))
# print(estimated_train_out)
print('nmse train')
print(NMSE(sig_out, estimated_train_out, alpha))

fm_test = model.compute_feature_matrix_det(test_in, alpha, weights)
estimated_test_out = np.matmul(fm_test, beta)
print('nmse test')
print(NMSE(test_out, estimated_test_out, alpha))


print('[ridge-regularized regression]')
lamb = 0.3
beta, _, rank, _ = np.linalg.lstsq(fm.T.dot(fm) + lamb*np.identity(fm.shape[1]), fm.T.dot(sig_out))

# print('out')
estimated_train_out  = np.matmul(fm, beta)
# print(np.shape(estimated_train_out))
# print(estimated_train_out)
print('nmse train')
print(NMSE(sig_out, estimated_train_out, alpha))

estimated_test_out = np.matmul(fm_test, beta)
print('nmse test')
print(NMSE(test_out, estimated_test_out, alpha))
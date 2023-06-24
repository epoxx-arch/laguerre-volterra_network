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
from reservoir_laguerre_volterra_network_structure import RLVN
from laguerre_volterra_network_structure import LVN
import optimization_utilities
# Pyython std library
import math
import sys
# 3rd party
import numpy as np
    
#
train_in, train_out = read_io('./signals_and_systems/long_finite_train.csv')

print(f'Train len: {len(train_in)}')

## Configs
# Structure
L = 3; H = 2; Q = 2
Fs = 25
# Laguerre smoothing constant
alpha = 0.7
# Number of runs
ntimes = 10

print(f'(L,H,Q) = ({L},{H},{Q})')

# Define W and C 
W = np.array([[1, 1.5],[2, 2.5],[3, 3.5]])
C = np.array([0.5, 2, 3, 4, 5])
LVN_offset = 0.5
LVN_W = W.T
LVN_C = np.array([[2, 4],[3, 5]])

print(np.shape(W))
print(np.shape(C))
print(np.shape(LVN_W))
print(np.shape(LVN_C))
#

# LVN definition
LVN_model = LVN()
LVN_model.define_structure(L, H, Q, 1/Fs)
LVN_train_out = LVN_model.compute_output(train_in, alpha, LVN_W, LVN_C, LVN_offset, False)
LVN_train_nmse = optimization_utilities.NMSE(train_out, LVN_train_out, alpha)

# RLVN definition
RLVN_model = RLVN(L, H, Q, 1/Fs, False)
RLVN_model.set_connection_weights(W)
RLVN_model.set_polynomial_coefficients(C)
RLVN_train_out = RLVN_model.predict(train_in, alpha)
RLVN_train_nmse = optimization_utilities.NMSE(train_out, RLVN_train_out, alpha)


print(f'LVN train NMSE: {LVN_train_nmse}')
print(f'RLVN train NMSE: {RLVN_train_nmse}')
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
import data_handling
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
sig_in, sig_out = data_handling.read_io('./signals_and_systems/finite_order_train.csv')
fm = model.compute_feature_matrix(sig_in, alpha, weights_range)
print(fm)
print(np.linalg.matrix_rank(fm))
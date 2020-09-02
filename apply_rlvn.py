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

# train_in, train_out = read_io('./signals_and_systems/finite_order_train.csv')
# test_in, test_out = read_io('./signals_and_systems/finite_order_test.csv')
train_in, train_out = read_io('./signals_and_systems/large_finite_train.csv')
test_in, test_out = read_io('./signals_and_systems/large_finite_test.csv')


L = 5
H = 5
Q = 4
Fs = 25
wrange = 1
alpha = 0.44
ntimes = 30
model = RLVN(L,H,Q, Fs)

print('LVN-like (H maps from filter bank to hidden layer)')
train_errors = []
test_errors = []
extended = False
for _ in range(ntimes):
    model.randomize_weights(weights_range = wrange, extended_weights = extended)
    model.train(in_signal = train_in, out_signal = train_out, alpha = alpha, l2_regularization = False, extended_weights = extended)
    estimated_train_out = model.predict(in_signal = train_in, extended_weights = extended)
    estimated_test_out = model.predict(in_signal = test_in, extended_weights = extended)

    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    train_errors.append(nmse_train)
    test_errors.append(nmse_test)
    
print(f'NMSE train: {np.mean(train_errors)} {np.std(train_errors)}')
print(f'NMSE  test: {np.mean(test_errors)} {np.std(test_errors)}')


print('Extended (HQ maps from filter bank to hidden layer)')
train_errors = []
test_errors = []
extended = True
for _ in range(ntimes):
    model.randomize_weights(weights_range = wrange, extended_weights = extended)
    model.train(in_signal = train_in, out_signal = train_out, alpha = alpha, l2_regularization = False, extended_weights = extended)
    
    estimated_train_out = model.predict(in_signal = train_in, extended_weights = extended)
    estimated_test_out =  model.predict(in_signal = test_in, extended_weights = extended)
    
    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    train_errors.append(nmse_train)
    test_errors.append(nmse_test)
        
print(f'NMSE train: {np.mean(train_errors)} {np.std(train_errors)}')
print(f'NMSE  test: {np.mean(test_errors)} {np.std(test_errors)}')

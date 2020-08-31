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
H = 3
Q = 4
Fs = 25
weights_range = 0.01
alpha = 0.44
ntimes = 30
model = RLVN(L,H,Q, Fs)

print('LVN-like')
train_errors = []
test_errors = []
for _ in range(ntimes):
    model.randomize_weights(weights_range)
    model.train(train_in, train_out, alpha, False)
    estimated_train_out = model.predict(train_in)
    estimated_test_out = model.predict(test_in)

    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    train_errors.append(nmse_train)
    test_errors.append(nmse_test)
        
print(f'NMSE train: {np.mean(train_errors)} {np.std(train_errors)}')
print(f'NMSE  test: {np.mean(test_errors)} {np.std(test_errors)}')

print('Extended')
train_errors = []
test_errors = []
for _ in range(ntimes):
    model.randomize_weights_ext(weights_range)
    model.train_ext(train_in, train_out, alpha, False)
    
    estimated_train_out = model.predict_ext(train_in)
    estimated_test_out =  model.predict_ext(test_in)
    
    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    train_errors.append(nmse_train)
    test_errors.append(nmse_test)
        
print(f'NMSE train: {np.mean(train_errors)} {np.std(train_errors)}')
print(f'NMSE  test: {np.mean(test_errors)} {np.std(test_errors)}')

# model.train(train_in, train_out, alpha, True)
# estimated_train_out = model.predict(train_in)
# estimated_test_out = model.predict(test_in)

# print('ridge nmse train')
# print(NMSE(train_out, estimated_train_out, alpha))
# print('ridge nmse test')
# print(NMSE(test_out, estimated_test_out, alpha))

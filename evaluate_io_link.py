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
import sys
# 3rd party
import numpy as np

if len(sys.argv) != 3:
    print('Error, use as ' + sys.argv[0] + ' \'short\'|\'long\ \'finite\'|\'infinite\'')
    exit(-1)
    
size = (sys.argv[1]).lower()
order = (sys.argv[2]).lower()
if size != 'short' and size != 'long':
    print('Error, size must be \'short\' or \'long\'')
    exit(-1)
if order != 'finite' and order != 'infinite':
    print('Erro r, order must be \'finite\' or \'infinite\'')
    exit(-1)    
    
print(f'Size = {size}, order = {order}')
    
if size == 'short':
    train_in, train_out = read_io('./signals_and_systems/' + order + '_order_train.csv')
    test_in, test_out   = read_io('./signals_and_systems/' + order + '_order_test.csv')
else:        # size == 'long'
    train_in, train_out = read_io('./signals_and_systems/long_' + order + '_train.csv')
    test_in, test_out   = read_io('./signals_and_systems/long_' + order + '_test.csv')

## Configs
# Structure
L = 3; H = 10; Q = 4
Fs = 25
# Abs range of random weights
wrange = 0.1
# Laguerre smoothing constant
alpha = 0.7
# Number of runs
seed = None
ntimes = 100
l2_regularization = False
if l2_regularization:
    print('\nUSING L2 REGULARIZATION\n')

print(f'(L,H,Q) = ({L},{H},{Q})')

# RLVN with H maps from filter bank to hidden layer (same as original LVN)
# Keep results with or without input-output direct link
iof_train_errors = []
iot_train_errors = []
iof_test_errors = []
iot_test_errors = []

for _ in range(ntimes):
    # All models use the same seed in a run
    #seed = np.random.randint(1000)
    
    # Without io / with bo links
    io_link = False
    # Train model and predict outputs
    model = RLVN(L, H, Q, Fs, io_link)
    model.randomize_weights(weights_range = wrange, seed = seed)
    model.train(in_signal = train_in, out_signal = train_out, alpha = alpha, l2_regularization = l2_regularization)
    estimated_train_out = model.predict(in_signal = train_in)
    estimated_test_out = model.predict(in_signal = test_in)
    # Comput and keep errors
    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    iof_train_errors.append(nmse_train)
    iof_test_errors.append(nmse_test)
    
    # With io / bo links
    io_link = True  
    # Train model and predict outputs
    model = RLVN(L, H, Q, Fs, io_link)
    model.randomize_weights(weights_range = wrange, seed = seed)
    model.train(in_signal = train_in, out_signal = train_out, alpha = alpha, l2_regularization = l2_regularization)
    estimated_train_out = model.predict(in_signal = train_in)
    estimated_test_out = model.predict(in_signal = test_in)
    # Comput and keep errors
    nmse_train = NMSE(train_out, estimated_train_out, alpha)
    nmse_test = NMSE(test_out, estimated_test_out, alpha)
    iot_train_errors.append(nmse_train)
    iot_test_errors.append(nmse_test)
    
print('Train')
print(f'NMSE without io/with bo: {np.mean(iof_train_errors)} ({np.std(iof_train_errors)})')
print(f'NMSE with io/bo:         {np.mean(iot_train_errors)} ({np.std(iot_train_errors)})')
print('Test')
print(f'NMSE without io/with bo: {np.mean(iof_test_errors)} ({np.std(iof_test_errors)})')
print(f'NMSE with io/bo:         {np.mean(iot_test_errors)} ({np.std(iot_test_errors)})')
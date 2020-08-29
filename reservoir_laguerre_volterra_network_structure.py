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

import numpy as np
import math

class RLVN:
    ''' Reservoir Laguerre-Volterra network '''
    
    def __init__(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        ''' Constructor. '''
        
        # Sanity check
        if any([ (not(isinstance(param,int)) or param <= 0) for param in [laguerre_order, num_hidden_units, polynomial_order, sampling_interval]]):
            print('Error, structural parameters must be positive integers.')
            exit(-1)
            
        # Structural parameters
        self.L = laguerre_order         # laguerre_order
        self.H = num_hidden_units       # num_hidden_units
        self.Q = polynomial_order       # polynomial_order
        self.T = sampling_interval      # sampling_interval
        
        
    def propagate_laguerre_filterbank(self, signal, alpha):
        ''' Propagate input signal through the Laguerre filter bank.
            The output is an (L,N) matrix '''
        
        # Sanity check
        if alpha <= 0:
            print('Error, alpha must be positive')
            exit(-1)
        if len(signal) == 0:
            print('Error, signal must be nonempty.')
            exit(-1)
        
        alpha_sqrt = math.sqrt(alpha)
        bank_outputs = np.zeros((self.L, 1 + len(signal)))      # The bank_outputs matrix initially has one extra column to represent zero values at n = -1
        
        # Propagate V_{j} with j = 0
        for n, sample in enumerate(signal):
            bank_outputs[0, n + 1] = alpha_sqrt * bank_outputs[0, n - 1 + 1] +  self.T * np.sqrt(1 - alpha) * sample
        
        # Propagate V_{j} with j = 1, .., L-1
        for j in range(1, self.L):
            for n in range(len(signal)):
                bank_outputs[j, n + 1] = alpha_sqrt * (bank_outputs[j, n - 1 + 1] + bank_outputs[j - 1, n + 1]) - bank_outputs[j - 1, n - 1  + 1]
        
        bank_outputs = np.array(bank_outputs[:,1:])
        # print(bank_outputs)
        
        return bank_outputs
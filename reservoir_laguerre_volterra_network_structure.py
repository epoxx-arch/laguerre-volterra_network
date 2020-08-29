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
        
        alpha_sqrt = math.sqrt(alpha)
        bank_outputs = np.zeros((self.L, 1 + len(signal)))      # The bank_outputs matrix initially has one extra column to represent zero values at n = -1
        
        # Propagate V_{j} with j = 0
        for n, sample in enumerate(signal):
            bank_outputs[0, n + 1] = alpha_sqrt * bank_outputs[0, n - 1 + 1] +  self.T * np.sqrt(1 - alpha) * sample
        
        # Propagate V_{j} with j = 1, .., L-1
        for j in range(1, self.L):
            for n in range(len(signal)):
                bank_outputs[j, n + 1] = alpha_sqrt * (bank_outputs[j, n - 1 + 1] + bank_outputs[j - 1, n + 1]) - bank_outputs[j - 1, n - 1  + 1]
        
        bank_outputs = bank_outputs[:,1:]
        
        return bank_outputs
        
        
    def compute_feature_matrix_rand(self, signal, alpha, weights_range):
        ''' Generates random weights within the specified range, linearly computes hidden units inputs and then compute the separated "coefficientless" polynomial outputs.'''
        
        # Sanity check
        if alpha <= 0:
            print('Error, alpha must be positive')
            exit(-1)
        if len(signal) == 0:
            print('Error, signal must be nonempty.')
            exit(-1)
            
        # Propagation through Laguerre filter bank returns an (L,N) matrix
        N = len(signal)
        laguerre_outputs = self.propagate_laguerre_filterbank(signal, alpha)
        # print('Laguerre')
        # print(np.shape(laguerre_outputs))
        # print(laguerre_outputs)
        
        # Random weights are (L, H) in |rand| < |weights_range|
        random_weights = (np.random.rand(self.L, self.H) * 2 * weights_range) - weights_range
        # print('Weights')
        # print(np.shape(random_weights))
        # print(random_weights)
        
        # The input of each hidden node at some moment is the dot product between a random vector and the outputs of the Laguerre bank
        hidden_nodes_in = np.matmul(laguerre_outputs.T, random_weights)
        # print('Nodes input')
        # print(np.shape(hidden_nodes_in))
        # print(hidden_nodes_in)
        
        # The feature matrix is (N, HQ+1), containing Q polynomial maps for each hidden node without coefficients
        feature_matrix = np.ones((N, self.H * self.Q + 1))
        for q in range(1, self.Q + 1):
            #print(np.shape(feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] ))
            feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] = np.power(hidden_nodes_in, q)
        
        # print('Feature matrix')
        # print(np.shape(feature_matrix))
        # print(feature_matrix)
        
        return random_weights, feature_matrix
        
    def compute_feature_matrix_det(self, signal, alpha, random_weights):
        ''' With given random weights, linearly computes hidden units inputs and then compute the separated "coefficientless" polynomial outputs.'''
        
        # Sanity check
        if alpha <= 0:
            print('Error, alpha must be positive')
            exit(-1)
        if len(signal) == 0:
            print('Error, signal must be nonempty.')
            exit(-1)
            
        # Propagation through Laguerre filter bank returns an (L,N) matrix
        N = len(signal)
        laguerre_outputs = self.propagate_laguerre_filterbank(signal, alpha)
        # print('Laguerre')
        # print(np.shape(laguerre_outputs))
        # print(laguerre_outputs)
        
        
        # The input of each hidden node at some moment is the dot product between a random vector and the outputs of the Laguerre bank
        hidden_nodes_in = np.matmul(laguerre_outputs.T, random_weights)
        # print('Nodes input')
        # print(np.shape(hidden_nodes_in))
        # print(hidden_nodes_in)
        
        # The feature matrix is (N, HQ+1), containing Q polynomial maps for each hidden node without coefficients
        feature_matrix = np.ones((N, self.H * self.Q + 1))
        for q in range(1, self.Q + 1):
            #print(np.shape(feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] ))
            feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] = np.power(hidden_nodes_in, q)
        
        # print('Feature matrix')
        # print(np.shape(feature_matrix))
        # print(feature_matrix)
        
        return feature_matrix
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

# 3rd party
import numpy as np
# Pyython std library
import math
from collections.abc import Iterable

class RLVN:
    ''' Reservoir Laguerre-Volterra network '''
    
    def __init__(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval, io_link):
        ''' Constructor '''
        
        # Sanity check
        if any([ (not(isinstance(param,int)) or param <= 0) for param in [laguerre_order, num_hidden_units, polynomial_order, sampling_interval]]):
            print('Error, structural parameters (L, H, Q and Fs) must be positive integers.')
            exit(-1)
        if any([ not isinstance(param,bool) for param in [io_link]]):
            print('Flags (extended_weight and io_link) must be booleans.')
            exit(-1)
            
        # Structural parameters
        self.L = laguerre_order         # laguerre order
        self.H = num_hidden_units       # number of hidden units
        self.Q = polynomial_order       # polynomial order
        self.T = sampling_interval      # sampling interval
        
        # Flags
        self.io_link = io_link
        
        # Random weights matrix
        self.random_weights = None
        # Least square solution
        self.ls_solution = None
        # The Laguerre alpha used to train the model is kept
        self.train_alpha = None 
    
    def propagate_laguerre_filterbank(self, signal, alpha):
        ''' Propagate input signal through the Laguerre filter bank.
            The output is an (L,N) matrix. '''
        
        # Sanity check
        if not isinstance(signal, Iterable):
            print('Error, input signal must be an iterable object')
            exit(-1)
        if alpha <= 0:
            print('Error, alpha must be positive')
            exit(-1)
        
        # Compute sqrt(alpha) a single time
        alpha_sqrt = math.sqrt(alpha)
        # The bank_outputs matrix initially has one extra column to represent zero values at n = -1
        bank_outputs = np.zeros((self.L, 1 + len(signal)))
        
        # Propagate V_{j} with j = 0
        for n, sample in enumerate(signal):
            bank_outputs[0, n + 1] = alpha_sqrt * bank_outputs[0, n - 1 + 1] +  self.T * np.sqrt(1 - alpha) * sample
        
        # Propagate V_{j} with j = 1, .., L-1
        for j in range(1, self.L):
            for n in range(len(signal)):
                bank_outputs[j, n + 1] = alpha_sqrt * (bank_outputs[j, n - 1 + 1] + bank_outputs[j - 1, n + 1]) - bank_outputs[j - 1, n - 1  + 1]
        
        bank_outputs = bank_outputs[:,1:]
        
        return bank_outputs
    
    def set_connection_weights(self, connection_weights):
        '''  '''
        if np.shape(connection_weights) != (self.L, self.H):
            print('Error, connection weights must be (L, H)')
            exit(-1)
        
        self.connection_weights = connection_weights
      
    def set_polynomial_coefficients(self, polynomial_coefficients):
        '''  '''
        
        
    
        self.polynomial_coefficients = polynomial_coefficients
        
    def compute_enhanced_input(self, signal, alpha):
        ''' Given an input signal, linearly computes hidden units inputs and then compute the polynomial outputs. '''
        
        if not isinstance(self.connection_weights, Iterable):
            print('Error, feed weights to the model before computing the enhanced input matrix for some signal.')
            exit(-1)
        if alpha <= 0:
            print('Error, alpha must be a strictly positive number')
            exit(-1)
        
        # Propagation through Laguerre filter bank returns an (L,N) matrix
        N = len(signal)
        laguerre_outputs = self.propagate_laguerre_filterbank(signal, alpha)
        
        # The input of each hidden node at some moment is the dot product between a random vector and the outputs of the Laguerre bank
        # Hidden nodes input matrix is (N,H)  
        hidden_nodes_in = laguerre_outputs.T @ self.connection_weights 
        
        # The enhanced input matrix is  populated by polynomial maps for each hidden node
            
        
            enhanced_input = np.ones((N, self.H * self.Q + 1))
            first_index = 1
        
        # All positions of the first column remain ones to account for the output offset
        # W = (L, H)
        for q in range(first_index, self.Q + 1):
            enhanced_input[:, 1  + (q - first_index) * self.H : 1 + (q - first_index + 1) * self.H] = np.power(hidden_nodes_in, q)

            
        
        return enhanced_input
        

    def predict(self, in_signal, alpha):
        ''' Predicts output signal using the current least squares solution. '''
        
        # Sanity check
        if not isinstance(self.polynomial_coefficients, Iterable):
            print('Error, train the model to define a least squares solution before predicting outputs.')
            exit(-1)
        if not isinstance(in_signal, Iterable):
            print('Error, input signal must be an iterable object')
            exit(-1)
        
        enhanced_input = self.compute_enhanced_input(signal=in_signal, alpha=alpha)
        out_signal = enhanced_input @ self.polynomial_coefficients
        
        return out_signal
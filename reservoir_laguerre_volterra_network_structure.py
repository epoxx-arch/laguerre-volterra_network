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
from collections.abc import Iterable

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
    
    
    def randomize_weights(self, weights_range, extended_weights):
        ''' Random weights are (L, H) or (L, HQ) depending upon extended_weights, in |rand| < |weights_range|. '''
        # Sanity check
        if isinstance(weights_range, Iterable):
            print('Error, range must be a scalar')
            exit(-1)
        if weights_range == 0:
            print('Error range must be nonzero')
            exit(-1)
        if not isinstance(extended_weights, bool):
            print('Error, extended_weights must be a boolean')
            exit(-1)
        
        # Randomize weights
        if extended_weights:    # W = (L, HQ)
            self.random_weights = (np.random.rand(self.L, self.H * self.Q) * 2 * weights_range) - weights_range
        else:                   # W = (L, H)
            self.random_weights = (np.random.rand(self.L, self.H) * 2 * weights_range) - weights_range
        
        
    def compute_feature_matrix(self, signal, alpha, extended_weights):
        ''' Given an in-signal, linearly computes hidden units inputs and then compute the separated "coefficientless" polynomial outputs.'''
        
        if not isinstance(self.random_weights, Iterable):
            print('Error, randomize weights before computing the feature matrix for some signal.')
            exit(-1)
        if not isinstance(extended_weights, bool):
            print('Error, extended_weights must be a boolean')
            exit(-1)
         
        # Propagation through Laguerre filter bank returns an (L,N) matrix
        N = len(signal)
        laguerre_outputs = self.propagate_laguerre_filterbank(signal, alpha)
        # print('Laguerre')
        # print(np.shape(laguerre_outputs))
        # print(laguerre_outputs)
        
        # The input of each hidden node at some moment is the dot product between a random vector and the outputs of the Laguerre bank
        # Hidden nodes input matrix is (N,H)
        hidden_nodes_in = laguerre_outputs.T @ self.random_weights
        # print('Nodes input')
        # print(np.shape(hidden_nodes_in))
        # print(hidden_nodes_in)
        
        # The feature matrix is (N, HQ+1), containing Q polynomial maps for each hidden node without coefficients
        feature_matrix = np.ones((N, self.H * self.Q + 1))
        
        # W = (L, HQ)
        if extended_weights:
            for q in range(1, self.Q + 1):
                feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] = np.power(hidden_nodes_in[:,(q - 1) * self.H : q * self.H], q)
        
        # W = (L, H)
        else:                       
            for q in range(1, self.Q + 1):
                feature_matrix[:, 1  + (q - 1) * self.H : 1 + q * self.H] = np.power(hidden_nodes_in, q)
            
        # print('Feature matrix')
        # print(np.shape(feature_matrix))
        # print(feature_matrix)
        
        return feature_matrix
        
        
        
    def train(self, in_signal, out_signal, alpha, l2_regularization, extended_weights):
        ''' Computes nonlinear random feature matrix from input signal and estimates a linear map to the output signal.
            It is possible to consider L2-regularization, performing Ridge regression. '''
        
        # Sanity checks
        if not(isinstance(in_signal, Iterable) and isinstance(out_signal, Iterable)):
            print('Error, both input and output signals must be iterable objects.')
            exit(-1)
        if len(in_signal) != len(out_signal):
            print('Error, length of input and output signals must be equal.')
            exit(-1)
        if not isinstance(l2_regularization, bool):
            print('Error, L2 regularization must be a boolean.')
            exit(-1)
            
        # Compute nonlinear feature matrix with the randomized weights
        feature_matrix = self.compute_feature_matrix(signal=in_signal, alpha=alpha, extended_weights=extended_weights)
        # Keep alpha used to train the model (only after the function calls that verify alpha values)
        self.train_alpha = alpha
        
        if l2_regularization:
            lamb = 1
            diagonal_ridge = lamb*np.identity(feature_matrix.shape[1])
            diagonal_ridge[0,0] = 1
            beta, _, rank, _ = np.linalg.lstsq( feature_matrix.T @ feature_matrix + diagonal_ridge,
                                                feature_matrix.T @ out_signal, rcond=None)
           
            # # Normal equation 
            # beta, _, rank, _ = np.linalg.lstsq( feature_matrix.T.dot(feature_matrix), feature_matrix.T.dot(out_signal), rcond=None)
            # print('ridge')
            # print(np.shape(feature_matrix.T @ feature_matrix))
            # print(np.linalg.matrix_rank(feature_matrix.T @ feature_matrix))
        else:
            beta, _, rank, _ = np.linalg.lstsq(feature_matrix, out_signal, rcond=None)
            # print('unreg')
            # print(np.shape(feature_matrix))
            # print(np.linalg.matrix_rank(feature_matrix))
        self.ls_solution = beta
        
        # print('sol')
        # print(self.ls_solution)
    
        
    def predict(self, in_signal, extended_weights):
        ''' Predicts output signal using the current least squares solution. '''
        
        # Sanity check
        if not isinstance(self.ls_solution, Iterable) or self.train_alpha == None:
            print('Error, train the model to define a least squares solution before predicting outputs.')
            exit(-1)
        if not isinstance(in_signal, Iterable):
            print('Error, input signal must be an iterable object')
            exit(-1)
            
        feature_matrix = self.compute_feature_matrix(signal=in_signal, alpha=self.train_alpha, extended_weights=extended_weights)
        out_signal = feature_matrix @ self.ls_solution
        
        return out_signal        
    
    
    ## Getters
    def get_weights(self):
        ''' Return the current randomized weights matrix. '''
        return self.random_weights
        
    def get_solution(self):
        ''' Return the current least_squares solution. '''
        return self.ls_solution
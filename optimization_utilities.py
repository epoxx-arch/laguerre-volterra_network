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
# 3rd party
import numpy as np
# Pyython std library
import math
from collections.abc import Iterable

# Normalized mean squared error
def NMSE(y, y_pred, alpha):
    if len(y) != len(y_pred):
        print("Actual and predicted y have different lengths")
        exit(-1)
    
    # Laguerre alpha paremeter determines the system memory (find reference for formula)
    M = laguerre_volterra_network_structure.laguerre_filter_memory(alpha)
    
    if len(y) <= M:
        print("Data length is less than required by the alpha parameter")
        exit(-1)
    
    y = np.array(y)
    y_pred = np.array(y_pred)
    error = y[M:] - y_pred[M:]
    
    NMSE = sum( error**2 ) / sum( y[M:]**2 )
    
    return NMSE
    
# Normalized mean squared error
def NMSE_explicit_memory(y, y_pred, M):
    if len(y) != len(y_pred):
        print("Actual and predicted y have different lengths")
        exit(-1)
 
    if len(y) <= M:
        print("Data length is less than required by the memory parameter")
        exit(-1)
    
    y = np.array(y)
    y_pred = np.array(y_pred)
    error = y[M:] - y_pred[M:]
    
    NMSE = sum( error**2 ) / sum( y[M:]**2 )
    
    return NMSE

# Break flat list-like solution into [alpha, W, C, offset] for a given LVN structure
# USED WITH METAHEURISTICS
def decode_solution(candidate_solution, L, H, Q):
    # Identify solution members
    alpha = candidate_solution[0]
    flat_W = candidate_solution[1 : (H * L + 1)]
    flat_C = candidate_solution[(H * L + 1) : (H * L + 1) + H * Q]
    offset = candidate_solution[(H * L + 1) + H * Q]
    
    # unflatten W and C
    W = []
    C = []
    for hidden_unit in range(H):
        W.append( flat_W[hidden_unit * L : (hidden_unit + 1) * L] )
        C.append( flat_C[hidden_unit * Q : (hidden_unit + 1) * Q] )
    
    return alpha, W, C, offset
    

#    
def randomize_weights(weights_range, L, H):
    ''' Random weights are (L, H) in |rand| < |weights_range|. '''

    # Sanity check
    if isinstance(weights_range, Iterable):
        print('Error, range must be a scalar')
        exit(-1)
    if weights_range == 0:
        print('Error range must be nonzero')
        exit(-1)

    # L and H define the 
    vec_cardinality = L
    num_hidden_units = H

    # Randomize weights; W = (cardinality, H)
    random_weights = (np.random.rand(vec_cardinality, num_hidden_units) * 2 * weights_range) - weights_range

    return random_weights

#
def train_poly_least_squares(rlvn_model, in_signal, out_signal, alpha):
    ''' Computes enhanced input matrix from input signal and estimates a linear map to the output signal.
    It is possible to consider L2-regularization, performing Ridge regression. '''
    
    # Sanity checks
    if not(isinstance(in_signal, Iterable) and isinstance(out_signal, Iterable)):
        print('Error, both input and output signals must be iterable objects.')
        exit(-1)
    if len(in_signal) != len(out_signal):
        print('Error, length of input and output signals must be equal.')
        exit(-1)
    
    # Compute enhanced input matrix
    enhanced_input = rlvn_model.compute_enhanced_input(signal=in_signal, alpha=alpha)
    
    # Verify rank of the enhanced input matrix 
    rank = np.linalg.matrix_rank(enhanced_input)
    if rank != np.shape(enhanced_input[1]):
        print('RANK DEFICIENCY')
    print(f'Cols = {np.shape(enhanced_input[1])}, rank = {rank}')

    # 
    l2_regularization = True
    if l2_regularization:
        lamb = 1e-1
        diagonal_ridge = lamb * np.identity(enhanced_input.shape[1])
        diagonal_ridge[0,0] = 0           
        poly_coefficients, _, _, _ = np.linalg.lstsq(enhanced_input.T @ enhanced_input + diagonal_ridge,
                                           enhanced_input.T @ out_signal, rcond=None)
                                           
        #poly_coefficients, _, rank, _ = np.linalg.lstsq(enhanced_input, out_signal, rcond=None)
    else:
        #poly_coefficients = np.linalg.pinv(enhanced_input.T @  ) @ out_signal
        poly_coefficients, _, _, _ = np.linalg.lstsq(enhanced_input, out_signal, rcond=None)
   
    return poly_coefficients  

    # Cost computation parameterized by the nesting function (define_cost)
    # modified_variable indicates which parameters were modified in the solution. -1 if all of them were.
    def compute_cost(candidate_solution, modified_variable):
        
        # IO
        train_input, train_output = data_handling.read_io(train_filename)

        # Get parameters from candidate solution
        alpha, W, C, offset = decode_solution(candidate_solution, L, H, Q)
        
        # If the weights were modified, set flag so LVN normalizes weights and scales coefficients before output computation 
        if modified_variable == -1 or (modified_variable >= 1 and modified_variable <= L * H):
            weights_modified = True
        else:
            weights_modified = False
            
        # Generate output and compute cost
        solution_system = laguerre_volterra_network_structure.LVN()
        solution_system.define_structure(L, H, Q, 1/Fs)
        solution_output = solution_system.compute_output(train_input, alpha, W, C, offset, weights_modified)
        
        cost = NMSE(train_output, solution_output, alpha)
        
        return cost
        
    return compute_cost


    
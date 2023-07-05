#!python3

# Copyright (C) 2023  Victor O. Costa

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
from laguerre_volterra_network import LVN
# 3rd party
import numpy as np
# Pyython std library
import math
from collections.abc import Iterable

# 
def laguerre_filter_memory(alpha):
    
    M = (-30 - math.log(1 - alpha)) / math.log(alpha)
    M = math.ceil(M)
    
    return M

# Normalized mean squared error
def NMSE(y, y_pred, alpha):
    if len(y) != len(y_pred):
        print("Actual and predicted y have different lengths")
        exit(-1)
    
    # Laguerre alpha paremeter determines the system memory (find reference for formula)
    M = laguerre_filter_memory(alpha)
    
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

# Break solution into [alpha, wrange] for a given LVN structure
def decode_alpha_range(candidate_solution):
    # Identify solution members
    alpha = candidate_solution[0]
    weights_range = candidate_solution[1]
    
    return alpha, weights_range

# Break flat list-like solution into [alpha, W] for a given LVN structure
def decode_alpha_weights(candidate_solution, L, H):
    # Identify solution members
    alpha = candidate_solution[0]
    flat_W = candidate_solution[1 : (H * L + 1)]
    
    # unflatten W
    W = []

    for bank_out in range(L):
        W.append( flat_W[bank_out * H : (bank_out + 1) * H] )
        
    return alpha, W

# Break flat list-like solution into [alpha, W, C] for a given LVN structure
def decode_alpha_weights_coefficients(candidate_solution, L, H, Q, bo_link):
    
    # Identify solution members
    alpha = candidate_solution[0]
    flat_W = candidate_solution[1 : (H * L + 1)]
    
    if bo_link:
        flat_C = candidate_solution[(H * L + 1) : (H * L + 1) + H * (Q - 1) + L + 1]
    else:
        flat_C = candidate_solution[(H * L + 1) : (H * L + 1) + H * Q  + 1]
    
    # unflatten W
    W = []
    for bank_out in range(L):
        W.append( flat_W[bank_out * H : (bank_out + 1) * H] )

    return alpha, W, flat_C

#    
def randomize_weights(weights_range, L, H):
    ''' Random weights are (L, H) in |rand| < |weights_range|. '''

    # Sanity check
    if isinstance(weights_range, Iterable):
        print('Error, range must be a scalar')
        exit(-1)
    if weights_range == 0:
        print('Error, range must be nonzero')
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
    # if rank != np.shape(enhanced_input[1]):
        # print('RANK DEFICIENCY')
    # print(f'Cols = {np.shape(enhanced_input[1])}, rank = {rank}')

    # 
    l2_regularization = True
    if l2_regularization:
        lamb = 1e-1
        diagonal_ridge = lamb * np.identity(enhanced_input.shape[1])
        diagonal_ridge[0,0] = 0           
        poly_coefficients, _, _, _ = np.linalg.lstsq(enhanced_input.T @ enhanced_input + diagonal_ridge,
                                           enhanced_input.T @ out_signal, rcond=None)
    else:
        # poly_coefficients = np.linalg.pinv(enhanced_input) @ out_signal
        poly_coefficients, _, _, _ = np.linalg.lstsq(enhanced_input, out_signal, rcond=None)
   
    return poly_coefficients  

# Compute cost of candidate solution, which is encoded as a flat array
# The content of this array depends on the solution_encoding parameter
#
# For solution_encoding = 2: Optimize alphas, weights and coefficients
# For solution_encoding = 1: Optimize alphas and weights
# For solution_encoding = 0: Optimize alphas and weights range
#
# Candidate solution encoding:
#   -- 0: [alpha, weights_range]
#   -- 1: [alpha, W(0,0) ... W(L-1,H-1)]
#   -- 2 w/o BO link: [alpha, W(0,0) ... W(L-1,H-1), C(0) ... C(H * Q + 1)]
#   -- 2 w/ BO link:  [alpha, W(0,0) ... W(L-1,H-1), C(0) ... C(H * (Q - 1) + L + 1)]
def define_cost(solution_encoding, L, H, Q, bo_link, Fs, train_filename):
    if solution_encoding < 0 or solution_encoding >= 3:
        print('Error, solution_encoding must be 0, 1 or 2')
        exit(-1)
      
    # IO
    train_in, train_out = data_handling.read_io(train_filename)

    # Cost computation parameterized by the nesting function (define_cost)
    def compute_cost(candidate_solution):
        # 
        if ((solution_encoding == 0 and len(candidate_solution) != 2) or
            (solution_encoding == 1 and len(candidate_solution) != H * L + 1) or
            (solution_encoding == 2 and bo_link and len(candidate_solution) != H * L + 1 + H * (Q - 1) + L + 1) or
            (solution_encoding == 2 and not bo_link and len(candidate_solution) != H * L + 1 + H * Q + 1)):
            
            print('Error, wrong length of the candidate solution for given solution encoding scheme')
            print(np.shape(candidate_solution))
            print(len(candidate_solution))
            print(H * L + 1)
            
            exit(-1)
        
        # LVN model
        candidate_model = LVN(L, H, Q, 1 / Fs, bo_link)
        
        # Get parameters from candidate solution, depending on the solution encoding scheme
        ## In solution encoding 0, weights are randomized and
        ##   poly coefficients are found with least-square errors
        if solution_encoding == 0:
            alpha, weights_range = decode_alpha_range(candidate_solution)
            W = randomize_weights(weights_range, L, H)
            
            # print(f'TEST: alpha is {alpha}, wrange is {weights_range}')
    
        ## In solution encoding 1, poly coefficients are found with least-square errors
        elif solution_encoding == 1:
            alpha, W = decode_alpha_weights(candidate_solution, L, H)
            
            # print(f'TEST: alpha is {alpha}, W is {W}')
        
        ## In solution encoding  2, all parameters are found with metaheuristics
        else:
            alpha, W, C = decode_alpha_weights_coefficients(candidate_solution, L, H, Q, bo_link)
            
            # print(f'TEST: alpha is {alpha}, W is {W},  C is {C}')
        
        # Feed weights to the model
        candidate_model.set_connection_weights(W)
        
        # Define C as least-squares solution for the matrix formulation
        if solution_encoding == 0 or solution_encoding == 1:
            C = train_poly_least_squares(candidate_model, train_in, train_out, alpha)
        
        # Set poly coefficients in the model
        candidate_model.set_polynomial_coefficients(C)

        # Predict
        ## Given signal, W and C, compute output signal and cost of the candidate solution
        
        # if solution_encoding == 2 and bo_link == True:
            # print(train_in)
        model_out = candidate_model.predict(train_in, alpha)
        cost = NMSE(train_out, model_out, alpha)
        
        return cost
        
    return compute_cost
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

# Python standard library
import math
# Third party
import numpy as np
# Own
from laguerre_volterra_network import LVN

# Simulate Laguerre-Volterra Network of arbitrary structure with randomized parameters
# Returns output signal and the parameters used, so that the same parameters can be used with the test set
def simulate_LVN_random_parameters(input_signal, L, H, Q, Fs, bo_link):
    # 
    alpha_range = 0.5
    alpha = np.random.uniform(0, alpha_range)  
    
    # 
    weights_range = 1
    W = (np.random.rand(L, H) * 2 * weights_range) - weights_range
    
    #
    if bo_link:
        C = np.random.rand(H * (Q - 1) + L + 1)
    else:
        C = np.random.rand(H * Q  + 1)
    
    #
    system = LVN(L, H, Q, 1 / Fs, bo_link)
    system.set_connection_weights(W)
    system.set_polynomial_coefficients(C)
    
    #
    output_signal = system.predict(input_signal, alpha)
    
    #
    system_parameters = {'alpha': alpha,
                         'W': W,
                         'C': C}
    
    return output_signal, system_parameters
    

# Simulate LVN of arbitrary structure with deterministic parameters, and return output signal
def simulate_LVN_deterministic(input_signal, L, H, Q, Fs, bo_link, parameters):
    alpha = parameters['alpha']
    W = parameters['W']
    C = parameters['C']
    
    system = LVN(L, H, Q, 1 / Fs, bo_link)
    system.set_connection_weights(W)
    system.set_polynomial_coefficients(C)
    
    output_signal = system.predict(input_signal, alpha)
    
    return output_signal


# Simulated infinite order (in Taylor and Volterra senses) system via a cascade of IIR filter and static nonlinearity
# The IIR filter is a sum of exponentially weighted moving averages (EWMAs)   
def simulate_cascaded_random(input_signal, num_ewmas):
    
    # Randomize betas
    betas = np.random.uniform(0.2, 0.8, num_ewmas)
    ewmas = [0 for _ in range(num_ewmas)]
    
    # 
    output_signal = []
    for data_i in range(len(input_signal)):
        # Compute sum of EWMAs
        ewmas_sum = 0
        for ewma_i in range(num_ewmas):
            ewmas[ewma_i] = (1 - betas[ewma_i]) * input_signal[data_i] + betas[ewma_i] * ewmas[ewma_i]
            ewmas_sum += ewmas[ewma_i]
        
        # Compute static nonlinearity
        y = math.exp(math.sin(ewmas_sum))
        output_signal.append(y)
    
    
    return output_signal, betas
    

#
def simulate_cascaded_deterministic(input_signal, betas):
    num_ewmas = len(betas)
    #
    ewmas = [0 for _ in range(num_ewmas)]
    # ewmas_inspect = [[0 for _ in range(num_ewmas)] for _ in range(len(input_signal))]
    #
    output_signal = []
    for data_i in range(len(input_signal)):
        # Compute sum of EWMAs
        ewmas_sum = 0
        for ewma_i in range(num_ewmas):
            ewmas[ewma_i] = (1 - betas[ewma_i]) * input_signal[data_i] + betas[ewma_i] * ewmas[ewma_i]
            ewmas_sum += ewmas[ewma_i]
            # ewmas_inspect[data_i][ewma_i] = ewmas[ewma_i]
        
        # Compute static nonlinearity
        y = math.exp(math.sin(ewmas_sum))
        output_signal.append(y)
    
    return output_signal
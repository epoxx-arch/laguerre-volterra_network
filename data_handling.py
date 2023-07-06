#!python3

# Copyright (C) 2023 Victor O. Costa

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
import csv
# Own
import simulated_systems
import optimization_utilities as ou

# Write a given LVN structure and system into a file 
def write_LVN_file(file_name, system_parameters, L, H, Q, Fs, bo_link):    
    alpha = system_parameters['alpha']
    W = system_parameters['W']
    C = system_parameters['C']
    
    lvn_filename = file_name + ".LVN"
    
    with open(lvn_filename, mode = 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([L, H, Q, Fs])
        writer.writerow([bo_link])
       
        writer.writerow([alpha])
        writer.writerow(W.flatten())
        writer.writerow(C)
    
# Reads LVN file and returns the system's parameters
def read_LVN_file(file_name):
    with open(file_name, mode = 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_strings = []
        for row in csv_reader:
            csv_strings.append(row)
        
        L, H, Q, Fs = list(np.array(csv_strings[0]).astype(int))
        print(type(L))
        
        bo_link = eval(csv_strings[1][0])
        
        alpha = float(csv_strings[2][0])
        flat_W = list(np.array(csv_strings[3]).astype(float))
        flat_C = list(np.array(csv_strings[4]).astype(float))
        
        concatenated_parameters = [alpha] + flat_W + flat_C
        alpha, W, C = ou.decode_alpha_weights_coefficients(concatenated_parameters, L, H, Q, bo_link)
        
        return alpha, W, C, L, H, Q, Fs, bo_link
        
        
def write_cascade_file(file_name, betas):
    cascade_filename = file_name + '.cascade'
    #
    with open(cascade_filename, mode = 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(betas)
    
    
def read_cascade_file(file_name):
    with open(file_name, mode = 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_strings = []
        for row in csv_reader:
            csv_strings.append(row)
    
    betas = list(np.array(csv_strings[0]).astype(float))
    
    return betas

    
# Generate IO data using a Gaussian White Noise (GWN) signal as input to enable the system to capture dynamics of frequency cross-terms, adding GWN to output to reach a certain SNR
def generate_io(system_type, num_samples, file_name, parameters):

    if system_type.lower() != "lvn" and system_type.lower() != "cascade":
        print("The system type must be \"lvn\" or \"cascade\"")
        exit(-1)
    
    # Unit is a zero mean and unit variance Gaussian white noise (GWN) signal
    input = np.random.normal(0.0, 1.0, num_samples)
    
    # Finite order
    if system_type == "lvn":
        L = 5; H = 3; Q = 4
        Fs = 25
        bo_link = False
        # Train
        if parameters == None:
            noiseless_output, parameters = simulated_systems.simulate_LVN_random_parameters(input, L, H, Q, Fs, bo_link)
        # Test
        else:
            noiseless_output = simulated_systems.simulate_LVN_deterministic_parameters(input, L, H, Q, Fs, bo_link, parameters)
        
        # Write LVN structure and parameters to a file
        write_LVN_file(file_name, parameters, L, H, Q, Fs, bo_link)
        
    # Infinite order
    else:
        # Number of IIR filters composing the cascade
        num_filters = 3
        # Train
        if parameters == None:
            noiseless_output, parameters = simulated_systems.simulate_cascaded_random(input, num_filters)
        # Test
        else:
            noiseless_output = simulated_systems.simulate_cascaded_deterministic(input, parameters)
        
        write_cascade_file(file_name, parameters)
        
    # Output additive Gaussian White Noise 
    ## Signal-to-Noise ratio in decibels
    SNR_db = 5                                         
    ## Average power of output signal    
    out_avg_pwr = np.mean(np.array(noiseless_output) ** 2)          
    out_avg_pwr_db = 10 * np.log10(out_avg_pwr)                           
    ## As SNR_db = sig_power_db - noise_power_db, noise_power_db = sig_power_db - SNR_db
    noise_avg_pwr_db = out_avg_pwr_db - SNR_db
    noise_avg_pwr = 10 ** (noise_avg_pwr_db / 10)
    ## For a GWN signal X, the average power is equal to the second moment E[X^2] = mean^2 + std^2.
    ## With zero mean, the average power is equal to std^2, the variance
    GWN_std = np.sqrt(noise_avg_pwr)
    noise = np.random.normal(0.0, GWN_std, num_samples)
    
    # Generate noisy output
    output = noiseless_output + noise
    
    # Write I/O file
    csv_name = file_name + ".csv"
    with open(csv_name, mode = 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(input)
        csv_writer.writerow(output)
    
    return parameters


# Read IO data from CSVs
def read_io(file_name):
    input = []
    output = []
    
    with open(file_name, mode = 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_strings = []
        for row in csv_reader:
            csv_strings.append(row)
        
        input_string = csv_strings[0]
        output_string = csv_strings[1]
        
        for index in range(len(input_string)):
            input.append(float(input_string[index]))
            output.append(float(output_string[index]))
        
    return input, output
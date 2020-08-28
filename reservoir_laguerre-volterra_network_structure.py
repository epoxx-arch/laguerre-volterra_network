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
    ''' .'''
    
    def __init__(self, laguerre_order, num_hidden_units, polynomial_order, sampling_interval):
        ''' Constructor. '''
        
        # Sanity check of structural parameters
        if any([ (not(is_integer(param)) or param <= 0) for param in [laguerre_order, num_hidden_units, polynomial_order, sampling_interval]]):
            print('Error, structural parameters must be positive integers.')
            exit(-1)
            
        # Structural parameters
        self.L = laguerre_order         # laguerre_order
        self.H = num_hidden_units       # num_hidden_units
        self.Q = polynomial_order       # polynomial_order
        self.T = sampling_interval      # sampling_interval
        
        
        
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

# Python standard library
import sys
# 3rd party
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Error, inform the statistic to be plotted')
    exit(-1)

stat = sys.argv[1]

if stat != 'mean' and stat != 'median' and stat != 'minimum':
    print('Error, statistic should be \'mean\', \'median\' or \'minimum\'')
    exit(-1)

if stat == 'mean':
    ylabel = 'Erro médio (EQMN)'
elif stat == 'median':
    ylabel = 'Erro mediano (EQMN)'
else: 
    ylabel = 'Erro mínimo (EQMN)'

function_evals = np.arange(100, 10100, 100)
ntimes = 30

optimization_strategies = ['0', '1', '2']
dataset_orders = ['finite', 'infinite']
dataset_partitions = ['train', 'test']
color_set = ['#ea5545', '#87bc45', '#27aeef']

for system_order in dataset_orders:
    for partition in dataset_partitions:
        plt.figure()
        for index, strategy in enumerate(optimization_strategies):
            base_filename   = './data/strategy_' + strategy + '_' + system_order + '_' + partition
            costs_matrix  = np.load(base_filename + '_costs.npy')
            
            # Plot average cost history for the given metaheuristic
            if stat == 'mean':
                cost_history_stat = np.mean(costs_matrix, axis=0)
            elif stat == 'median':
                cost_history_stat = np.median(costs_matrix, axis=0)
            else:
                cost_history_stat = np.min(costs_matrix, axis=0)
            
            
            plt.plot(function_evals, cost_history_stat, label='Estratégia ' + strategy, linewidth=3, color=color_set[index])

        plt.title(stat + '_' + partition + '_' + system_order)
        plt.xlabel('AFO', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.legend(fontsize=16)

        plt.show()
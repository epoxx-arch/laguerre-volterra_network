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

function_evals = np.arange(100, 10100, 100)
ntimes = 30

optimization_strategies = ['0', '1', '2']
dataset_orders = ['finite', 'infinite']
dataset_partitions = ['train', 'test']
color_set = ['#ea5545', '#87bc45', '#27aeef']

for partition in dataset_partitions:
    for system_order in dataset_orders:
        plt.figure()
        for index, strategy in enumerate(optimization_strategies):
            base_filename   = './data/strategy_' + strategy + '_' + system_order + '_' + partition
            costs_matrix  = np.load(base_filename + '_costs.npy')
            
            # Plot average cost history for the given metaheuristic
            average_cost_history = np.sum(costs_matrix, axis=0) / ntimes
            plt.plot(function_evals, average_cost_history, label='Estratégia ' + strategy, linewidth=3, color=color_set[index])

        plt.title(partition + '_partition_' + system_order + '_order')
        plt.xlabel('AFO', fontsize=18)
        plt.ylabel('Erro médio (EQMN)', fontsize=18)
        plt.legend(fontsize=16)

        plt.show()
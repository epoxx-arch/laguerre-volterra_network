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

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE+2)  # fontsize of the figure title

# Load data (y axis)
finite_train = np.load('data/wrange_finite_train_costs.npy')
finite_test = np.load('data/wrange_finite_test_costs.npy')
infinite_train = np.load('data/wrange_infinite_train_costs.npy')
infinite_test = np.load('data/wrange_finite_test_costs.npy')

# Geometric progression of ranges (x axis)
prog_start = 2**-15
prog_ratio = 2
prog_n = 31
weights_ranges = [prog_start * (prog_ratio ** i) for i in range(prog_n)]

#
fig1, axs1 = plt.subplots(2, sharex=True)
fig1.suptitle('Sistema de ordem finita')

axs1[0].plot(weights_ranges, finite_train, label = 'Dados de treino', linewidth = 2, color = 'k')
axs1[0].set_xscale('log', base = 2)
axs1[0].legend(loc="upper left")
# axs1[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
# axs1[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

axs1[1].plot(weights_ranges, finite_test, label = 'Dados de teste', linewidth = 2, color = 'k')
axs1[1].set_xscale('log', base = 2)
axs1[1].legend(loc="upper left")
# axs1[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
# axs1[1].xaxis.set_major_locator(ticker.MultipleLocator(1))

# Hidden subplot to set common labels
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel('NMSE')
plt.xlabel('Intervalos dos pesos de conexão (W)')
plt.show()

#
fig2, axs2 = plt.subplots(2, sharex=True)
fig2.suptitle('Sistema de ordem infinita')

axs2[0].plot(weights_ranges,infinite_train, label = 'Dados de treino', linewidth = 2, color = 'k')
axs2[0].set_xscale('log', base = 2)
axs2[0].legend(loc="upper left")
# axs2[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
# axs2[0].xaxis.set_major_locator(ticker.MultipleLocator(1))


axs2[1].plot(weights_ranges, infinite_test, label = 'Dados de teste', linewidth = 2, color = 'k')
axs2[1].set_xscale('log', base = 2)
axs2[1].legend(loc="upper left")
# axs2[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
# axs2[1].xaxis.set_major_locator(ticker.MultipleLocator(1))

# Hidden subplot to set common labels
fig2.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel('NMSE')
plt.xlabel('Intervalos dos pesos de conexão (W)')
plt.show()
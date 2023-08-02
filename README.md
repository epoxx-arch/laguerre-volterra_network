# Laguerre-Volterra networks

The Laguerre-Volterra network (LVN) is a Volterra-equivalent connectionist architecture, which combines a bank of discrete Laguerre filters and a layer of polynomial activation functions.
This architecture is designed to model nonlinear dynamic systems from input-output signals.
In this way, the network is trained using gradient-based methods or metaheuristics.

This repository implements the LVN in a way which is independent from the possible optimization methodologies.
It also implements a comparison between different strategies to train LVNs.

### Those strategies are:

[Strategy 0] Optimize ALPHA and RANGE of random weights w/ metaheuristics
Weights are randomized and polynomial coefficients are computed as a least-squares solution

[Strategy 1] Optimize ALPHA and W w/ metaheuristics
Polynomial coefficients are computed as a least-squares solution

[Strategy 2] Optimize ALPHA, W and C w/ metaheuristics

### Organization of the repository:

## Third party software dependencies
* Python 3.9.13
    * NumPy 1.21.2 (vector math)
    * Matplotlib 3.5.1 (plotting)
    
## List of modules
* base_metaheuristic.py
    + simulated_annealing.py
    + particle_swarm_optimization.py
    + ant_colony_for_continuous_domains.py
* laguerre_volterra_network.py
* optimization_utilities.py
* simulated_systems.py
* data_handling.py

## Scripts and their uses
* generate_datasets.py                    - Uses the data_handling module to generate synthetic train and test IO signals from simulated systems.
* optimization_examples.py                - Optimizes LVN using different strategies (mostly used for verification)
* strategy_comparison_collect_results.py  - Runs some specified metaheuristic 30 times and stores the solutions found, along with their errors on test signals
* strategy_comparison_plot_results.py     - Plots the search histories based on results from strategy_comparison_collect_results.py
* weights_ranges_collect_results.py       - Evaluates ranges of the weights W on the optimization strategy 1
* weights_ranges_plots.py                 - Plots the NMSE associated with different ranges of W, based on results from weights_ranges_collect_results.py
* evaluate_bo_link.py                     - Evaluates the impact of a structural change in the network, regarding linear terms
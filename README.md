# Reservoir Laguerre-Volterra networks

The Laguerre-Volterra network (LVN) is a Volterra-equivalent connectionist architecture, which combines a bank of discrete Laguerre filters and a layer of polynomial activation functions.
This architecture is designed to model nonlinear dynamic systems from input-output signals, and some of its parameters enter nonlinearly in the IO equations.
In this way, the network is trained using gradient-based methods or metaheuristics.

Here we modify the LVN architecture according to the resevoir computing paradigm, in which only parameters that enter the IO equation in a linear way are optimized using tools from linear algebra.

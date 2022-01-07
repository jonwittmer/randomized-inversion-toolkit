This repo contains tools for using randomized methods to solve linear inverse problems. 

## Code layout 
The code is broken down into 3 main sections: 
* src
* examples 
* utils

### src
This directory contains the `RandomizationStrategy` code along with solvers/optimizers that are compatible with randomization strategies. Each `RandomizationStrategy` is designed to be interchangeable in application code by just changing the strategy. The only method that should be called from application code is: 

    solution = RandomizationStrategy.solve()

This makes it easy for application code to change strategy. 

Currently, 2 solvers are supported: numpy's least squares solver and scipy's conjugate gradient solver. The wrapper classes provided are meant to provide a consistent interface to various solvers so that application code can change solvers with minimal code modification. 

### examples
Several examples are provided to demonstrate the effectiveness of various randomization strategies for solving linear inverse problems and to provide a basis for extension. An example of how to implement a randomized solver for nonlinear PDE constrained inverse problems is also provided using the [Hippylib](www.github.com/hippylib/hippylib) library. See the Hippylib documentation for installation instructions. All other examples will run with only `numpy`, `scipy`, and `matplotlib` installed. 

### utils
This directory provides a few helper functions related to plotting for the nonlinear Hippylib example. 

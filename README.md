# BayesOPT + Pandora
(Exotic) Bayesian Optimization with traditional policies (EI, UCB, Thompson Sampling) and Pandora Box-inspired policies (Gittins, Surrogate, UCB+EI, TS+EI) in JAX and GPJAX
Note: installing JAX and GPJAX packages require GPJAX 0.6.8, jaxlib==0.4.7 and jax>=0.4.9

## Gittins Computation
Two methods (built-in optimization method "BFGS" and bisection method) for computing Gittins index of n boxes given the mean and standard deviation of their Gaussian distributed values

## Plot by Iterations
Ploting Gaussian Process and acquisition functions in each iteration, as well as the change in regrets over iterations and stopping time
Note: this notebook is not compactible with GPJAX 0.6.8 + jaxlib 0.4.7, for compatible version, see Regrets Computation

## Regrets Computation
Computing and storing arrays of global optimum, best observed values, regrets and acquisiton functions over iterations for multiple instances with specified kernels

## Performance Comparison
Comparing performance (mean traditional regrets and cost-adjusted regrets across instances) of different policies


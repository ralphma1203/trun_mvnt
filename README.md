This module reproduces the Efficient sampling algorithm of truncated multivariate (scale) mixtures of normals 
under linear inequality constraints proposed by Li and Ghosh (2015) <doi:10.1080/15598608.2014.996690> under
Python environment (analogous to the R package tmvmixnorm).

The two cores function rtmvn() and rtmvt() are useful to overcoming difficulties in simulating truncated normal 
and Student's t distribution respectively. Efficient rejection sampling for simulating truncated univariate normal 
distribution is also included in the modeule, which shows superiority in terms of acceptance rate and numerical 
stability compared to existing methods. An efficient function for sampling from truncated multivariate normal 
distribution subject to convex polytope restriction regions based on Gibbs sampler.

Author: Ting Fung (Ralph) Ma
email: tingfung.ma@wisc.edu

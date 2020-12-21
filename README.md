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

Examples:

Suppose we would like to sample from multivariate normal $N(\mu, \Sigma)$ subject to the constraints

```{r}
from trun_mvnt import rtmvn, rtmvt

Mean = np.array([0,0])
Sigma = np.array([1, 0.5, 0.5, 1]).reshape((2,2))

D = np.array([1,0,0,1,1,-1]).reshape((3,2))
lower = np.array([-2,-1,-2])
upper = np.array([2,3,5])

```

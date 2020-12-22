output: rmarkdown::github_document

This module reproduces the Efficient sampling algorithm of truncated multivariate (scale) mixtures of normals 
under linear inequality constraints proposed by Li and Ghosh (2015) ([doi:10.1080/15598608.2014.996690](https://www.tandfonline.com/doi/abs/10.1080/15598608.2014.996690)) under
Python environment (analogous to the R package [tmvmixnorm](https://cran.r-project.org/web/packages/tmvmixnorm/index.html)).

The two cores function rtmvn() and rtmvt() are useful to overcoming difficulties in simulating truncated normal 
and Student's t distribution respectively. Efficient rejection sampling for simulating truncated univariate normal 
distribution is also included in the modeule, which shows superiority in terms of acceptance rate and numerical 
stability compared to existing methods. An efficient function for sampling from truncated multivariate normal 
distribution subject to convex polytope restriction regions based on Gibbs sampler.

author: Ting Fung (Ralph) Ma

email: tingfung.ma@wisc.edu (feel free to email me if you find any bug!)


Summary:

In short, this module can be used to generate random sample of truncated multivariate normal
(and Student't). Note that the truncation can be in the form of many (linear) constraints.

Suppose we want to draw sample from p-dimensioanl normal <img src="https://render.githubusercontent.com/render/math?math=X_{p\times%201}\sim N(\mu_{p\times%201}, \Sigma_{p\times%20p})">), and similarly for Student's t with df=<img src="https://render.githubusercontent.com/render/math?math=\nu">,

subject to constranst(s): ![formula](https://render.githubusercontent.com/render/math?math=l\leq%20DX\leq%20u&mode=inline),
where  ![formula](https://render.githubusercontent.com/render/math?math=D&mode=inline) is ![formula](https://render.githubusercontent.com/render/math?math=m\times%20p&mode=inline), even if ![formula](https://render.githubusercontent.com/render/math?math=m>p).


The core functions rtmvn() and rtmvt() can solve the problem well.



Examples:

```{[python}
import numpy as np
from trun_mvnt import rtmvn, rtmvt


########## Traditional problem  ##########
##### lower < X < upper #####
# So D = identity matrix

D = np.diag(np.ones(4))
lower = np.array([-1,-2,-3,-4])
upper = -lower
Mean = np.zeros(4)
Sigma = np.diag([1,2,3,4])

n = 10 # want 500 final sample
burn = 100 # burn-in first 100 iterates
thin = 1 # thinning for Gibbs


random_sample = rtmvn(n, Mean, Sigma, D, lower, upper, burn, thin) 
# Numpy array n-by-p as result!
random_sample

# Suppose you want a Student t

nu = 10
random_sample_t = rtmvt(n, Mean, Sigma, nu, D, lower, upper, burn, thin)
# Numpy array n-by-p as result!
random_sample_t

########## Non-full rank problem (more constraints than dimension) ##########
Mean = np.array([0,0])
Sigma = np.array([1, 0.5, 0.5, 1]).reshape((2,2)) # bivariate normal

D = np.array([1,0,0,1,1,-1]).reshape((3,2)) # non-full rank problem
lower = np.array([-2,-1,-2])
upper = np.array([2,3,5])

n = 500 # want 500 final sample
burn = 100 # burn-in first 100 iterates
thin = 1 # thinning for Gibbs

random_sample = rtmvn(n, Mean, Sigma, D, lower, upper, burn, thin) # Numpy array n-by-p as result!




```

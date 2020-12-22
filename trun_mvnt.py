
import math
import numpy as np

from scipy.stats import norm, weibull_min

##### Functions for acceptance rate #####
# Acceptance rate of normal rejection sampling
def norm_acc(a, b):
    return (norm.cdf(b) - norm.cdf(a))


# Acceptance rate of half-normal rejection sampling
def halfnorm_acc(a, b):
    return 2*norm_acc(a, b)

# Acceptance rate of uniform rejection sampling
def unif_acc(a,b):
    constant = math.sqrt(2*math.pi)/(b-a)
    cp = constant*norm_acc(a, b)
    
    if (a <= 0 and b >= 0):
        return cp
    
    if a > 0:
        return cp*math.exp(a**2/2)
    
# Acceptance rate of translated-exponential rejection sampling
def exp_acc_opt(a,b):
    lam = (a + math.sqrt(a**2 + 4))/2
    return math.sqrt(2*math.pi)*norm_acc(a, b)*lam \
        *math.exp(-lam**2/2 + lam*a)

##### Functions for rejection sampling #####
# Normal rejection sampling
def norm_rej(a, b=math.inf):
    
    acc = 0
    
    while 1:
        x = norm.rvs()
        acc += 1 # one more draw
        
        if x >= a and x <= b:
            d = dict()
            d['x'] = x ; d['acc'] = acc
            return d
    
# Half-normal rejection sampling
def halfnorm_rej(a, b):
    acc = 0
    
    while 1:
        x = norm.rvs()
        acc += 1 # one more draw
        
        if abs(x) >= a and abs(x) <= b:
            d = dict()
            d['x'] = abs(x); d['acc']   = acc
            return d

# Uniform rejection sampling
def unif_rej(a,b):
        
    if 0 >= a and 0 <= b: type_ab = '1'
    if a > 0: type_ab = '2'
    if b < 0: type_ab = '3'
    
    # different cases for the ratio
    def rho_fun(type_ab):
        return {
            '1': math.exp(-x**2/2),
            '2': math.exp(-(x**2 - a**2)/2),
            '3': math.exp(-(x**2 - b**2)/2)
            }[type_ab]
        
    acc = 0
    
    while 1:
        x = np.random.uniform(low=a, high=b, size=1)
        u = np.random.uniform()
        
        acc += 1
        
        rho = rho_fun(type_ab)
        if u <= rho:
            d = dict()
            d['x'] = x; d['acc'] = acc
            return d
        
# Translated-exponential rejection sampling
def exp_rej(a, b=math.inf, lam_type='default'):
    if lam_type == 'default':
        lam = a
    else:
        lam = (a + math.sqrt(a**2 + 4))/2
    
    # rho is continuous at lambda=a, 
    # so don't need to change the expression of rho
    
    acc = 0
    
    while 1:
        x = weibull_min.rvs(1, loc=0, scale=1/lam) + a
        u = np.random.uniform()
        rho = math.exp(-(x-lam)**2/2)
        
        acc += 1
        
        if u <= rho and x < b:
            d = dict()
            d['x'] = x; d['acc'] = acc
            return d


##### Functions for pdf/cdf of truncated univariate distribution #####

# Density function of truncated univariate normal distribution
def dtuvn(x, mean, sd, lower, upper):
    if x < lower or x > upper:
        return 0
    else:
        value = norm.pdf(x, mean, sd)
        value = value/(norm.cdf(upper, mean, sd) - norm.cdf(lower, mean, sd))
        return value

# Distribution function of truncated univariate normal distribution
def ptuvn(x, mean, sd, lower, upper):
    if x <= lower: return 0
    if x >= upper: return 1
    
    num = norm.cdf(x, mean, sd) - norm.cdf(lower, mean, sd)
    dem = norm.cdf(upper, mean, sd) - norm.cdf(lower, mean, sd)
    
    return num/dem


##### Rejection sampling of truncated univariate normal #####

# Rejection sampling of standardized truncated univariate normal distribution
def imp(a,b):
    
    def lowerb1(a):
        return (math.sqrt(math.pi/2)*math.exp(a**2/2) + a)
    
    # lower bound of b for normal vs unif in 0 in [a,b]
    def lowerb2(a):
        return (math.sqrt(2*math.pi) + a)
    
    # lower bound of b for exp vs unif in [a,b]>=0
    def lowerb(a):
        lam = (a + math.sqrt(a**2 + 4))/2
        value = a + math.exp(0.5)/lam \
            *math.exp(( a**2 - a*math.sqrt(a**2 + 4) )/4 )
        
        return value
    
    # Case 1: [a,infty)
    def imp_case1(a,b=math.inf):
        if a < 0:
            return norm_rej(a,b) 
        else:
            if a < 0.25696:
                return halfnorm_rej(a,b)
            else:
                return exp_rej(a,b, 'opt')
            
    # Case 2: 0 in [a,b], a<0<b
    def imp_case2(a,b):
        if b > lowerb2(a):
            return norm_rej(a,b)
        else:
            return unif_rej(a,b)
            
        
    # Case 3: [a,b], a>=0
    def imp_case3(a,b):
        if a <= 0.25696:
            blower1 = lowerb1(a)
            if b <= blower1:
                return unif_rej(a,b)
            else:
                return halfnorm_rej(a,b)
            
        else:
            blower2 = lowerb(a)
            if b <= blower2:
                return unif_rej(a,b)
            else:
                return exp_rej(a,b,'opt')
            
    # Case 4: (-infty,b] (symmetric to Case 1)
    def imp_case4(a,b):
        temp = imp_case1(-b, -a)
        x = -temp['x']
        
        d = dict()
        d['x'] = x; d['acc'] = temp['acc']
        return d
    
    # Case 5: [a,b], b<=0 (symmetric to Case 3)
    def imp_case5(a,b):
        temp = imp_case3(-b,-a)
        x = -temp['x']
        
        d = dict()
        d['x'] = x; d['acc'] = temp['acc']
        return d
    

    if a == -math.inf or b == math.inf:
        if b == math.inf:
            return imp_case1(a,b)
        else:
            return imp_case4(a, b)
    else:
        if a < 0 and b > 0 : return imp_case2(a,b)
        if a >= 0: return imp_case3(a,b)
        if b <= 0: return imp_case5(a,b)
        

# Acceptance rate of truncated univariate normal distribution rejection sampling
def imp_acc(a,b):
    def lowerb1(a):
        return math.sqrt(math.pi/2)*math.exp(a**2/2) + a
    
    # lower bound of b for normal vs unif in 0 in [a,b]
    def lowerb2(a):
        return math.sqrt(2*math.pi) + a
    
    # lower bound of b for exp vs unif in [a,b]>=0
    def lowerb(a):
        lam = (a + math.sqrt(a**2 + 4))/2
        value = a + math.exp(0.5)/lam \
            *math.exp(( a**2 - a*math.sqrt(a**2 + 4) )/4 )
        
        return value
  
    # Case 1: [a,infty)
    def imp_acc_case1(a,b):
        if a <0: 
            return norm_acc(a,b)
        else:
            if a < 0.25696:
                return halfnorm_acc(a,b)
            else:
                return exp_acc_opt(a,b)

    # Case 2: 0 in [a,b], a<0<b
    def imp_acc_case2(a,b):
        if b > lowerb2(a):
            return norm_acc(a,b)
        else:
            return unif_acc(a,b)
    
    # Case 3: [a,b], a>=0
    def imp_acc_case3(a,b):
        if a <= 0.25696:
            blower1 = lowerb1(a)
            if b <= blower1:
                return unif_acc(a,b)
            else:
                return halfnorm_acc(a,b)
        else:
            blower2 = lowerb(a)
            if b <= blower2:
                return unif_acc(a,b)
            else:
                return exp_acc_opt(a,b)
            
    # Case 4 and 5 are symmetric to 1 and 3
    
    if a == -math.inf or b == math.inf:
        if b == math.inf:
            return imp_acc_case1(a)
        else:
            return imp_acc_case1(-b)
    else:
        if a < 0 and b > 0: return imp_acc_case2(a,b)
        if a >= 0: return imp_acc_case3(a,b)
        if b <= 0: return imp_acc_case3(-b, -a)
        

# Random number generation for truncated univariate normal distribution
def rtuvn(n=1, mean=0, sd=1, lower=-math.inf, upper=math.inf):
    
    
    # transform the boundaries
    a = (lower - mean)/sd
    b = (upper - mean)/sd

    # generate n samples from TN(0,1;a,b)
    Z = np.empty(n)
    for i in range(n):
        temp = imp(a,b)
        Z[i-1] = temp['x']
        
    return sd*Z + mean


##### Core functions: Random number generation for truncated multivariate normal/t distribution #####

#' Random number generation for truncated multivariate normal distribution subject to linear inequality constraints
def rtmvn(n=1, Mean=np.zeros(1), Sigma=np.ones(1), \
          D=np.ones(1), \
          lower=np.array(-math.inf), upper=np.array(math.inf), \
          burn=10, thin=1, ini=[]):
    
    D = np.matrix(D); [m,p] = D.shape # m constraints p-dimensional TN
    Sigma = np.matrix(Sigma)
    
    Mean = np.matrix(Mean)
    lower = np.matrix(lower); upper = np.matrix(upper)
    
    
    if Sigma.shape[1] != p or Sigma.shape[0] != Sigma.shape[1]: # check D and Sigma dimension
        print('No. col of "D" should match with dim. of "Sigma", and "Sigma" should be square matrix')
        return []
    
    if Mean.size != p or lower.size != m or upper.size != m : # check Mean an D matrix dimension 
        print('dimension of "Mean should match with No. col of "D". Dimension of "lower" and "upper" should match with No. row of "D"!')
        
        return []
    else:
        Mean = Mean.reshape((p,1))
        lower = lower.reshape((m,1)); upper = upper.reshape((m,1))
    
    
    if Mean.shape == (1,1):
        if lower >= upper:
            print('"lower" should be smaller than "upper"!')
            return []
        else:
            return np.array(rtuvn(n, Mean, math.sqrt(Sigma[0]), lower, upper))
    else:
        
      
        if any(lower >= upper) : # check any lower >= upper
            print('lower bound must be smaller than upper bound!')
            return []
        
        bound_check = 0
        
        if np.array(ini).size == p :
            ini_test = np.matmul(D, ini)
            lower_log = ini_test >= (lower + 1e-8)
            upper_log = ini_test <= (upper - 1e-8)
            bound_check = np.prod(np.multiply(lower_log,upper_log))
               
            if bound_check == 0:
                print('initial is outside or too close from boundary, will be auto-corrected by MP-inverse!')
        elif bound_check == 0:
            ini = np.matmul(np.linalg.pinv(D), (lower + upper)/2)
            
    if any(np.array([burn,thin,n]) % 1 != 0) :
        print('burn, thin and n must be integer!\n')
        return []
    if any(np.array([burn,thin,n-1]) % 1 != 0):
        print("burn, thin must be non-negative interger, n must be positive integer\n")
    
    a = lower - np.matmul(D, Mean)
    b = upper - np.matmul(D, Mean)
    Sigma_chol = np.linalg.cholesky(Sigma)
    R = np.matmul(D, Sigma_chol)
    
    
    z = np.linalg.solve(Sigma_chol, ini - Mean)
    
    keep_x = np.zeros( ((thin+1)*n+burn, p) )

    
    for i in range( (thin+1)*n+burn ):
        for j in range(p):
            rj = R[:,j]           # the jth column of R
            Rj = np.delete(R,j,1) # m by p-1 matrix by removing the jth column of R
            zj = np.delete(z,j).reshape((p-1,1))  # p-1 vector by removing the jth element
            a_temp = a - np.matmul(Rj, zj)
            b_temp = b - np.matmul(Rj, zj)
            
            # ignoring rj = 0, as alwyays fulfill
            pos = np.array(rj > 0)
            neg = np.array(rj < 0)
            
            if pos.sum() == 0:
                lower_pos = -math.inf ; upper_pos = math.inf
            else:
                lower_pos = (a_temp[pos]/rj[pos]).max() # when r_jk>0, a associated with lower bound
                upper_pos = (b_temp[pos]/rj[pos]).min() #              b                 upper
           
            if neg.sum() == 0:
                upper_neg = math.inf; lower_neg = -math.inf
            else:
                upper_neg = (a_temp[neg]/rj[neg]).min() # when r_jk<0, a                 upper
                lower_neg = (b_temp[neg]/rj[neg]).max() #              b                 lower
            
            lower_j = max(lower_pos, lower_neg)
            upper_j = min(upper_pos, upper_neg)
            
            z[j] = rtuvn(n=1, mean=0, sd=1, lower=lower_j, upper=upper_j)
            
        keep_x[i,:] = (np.matmul(Sigma_chol,z) + Mean).reshape(1,p)
    
    final_ind = np.array(range((thin+1)*n))
    final_ind = final_ind[0:len(final_ind):(thin+1)] + thin + burn
    
    return keep_x[final_ind,:]


#' Random number generation for truncated multivariate Student's t distribution subject to linear inequality constraints
def rtmvt(n=1, Mean=np.zeros(1), Sigma=np.ones(1), nu=100, \
          D=np.ones(1), \
          lower=np.array(-math.inf), upper=np.array(math.inf), \
          burn=10, thin=1, ini=[]):
    D = np.matrix(D); [m,p] = D.shape # m constraints p-dimensional TN
    Sigma = np.matrix(Sigma)
    
    Mean = np.matrix(Mean)
    lower = np.matrix(lower); upper = np.matrix(upper)
    
    
    if Sigma.shape[1] != p or Sigma.shape[0] != Sigma.shape[1]: # check D and Sigma dimension
        print('No. col of "D" should match with dim. of "Sigma", and "Sigma" should be square matrix')
        return []
    
    if Mean.size != p or lower.size != m or upper.size != m : # check Mean an D matrix dimension 
        print('dimension of "Mean should match with No. col of "D". Dimension of "lower" and "upper" should match with No. row of "D"!')
        
        return []
    else:
        Mean = Mean.reshape((p,1))
        lower = lower.reshape((m,1)); upper = upper.reshape((m,1))
    
    
    if Mean.shape == (1,1):
        if lower >= upper:
            print('"lower" should be smaller than "upper"!')
            return []
        else:
            return np.array(rtuvn(n, Mean, math.sqrt(Sigma[0]), lower, upper))
    else:
        
      
        if any(lower >= upper) : # check any lower >= upper
            print('lower bound must be smaller than upper bound!')
            return []
        
        bound_check = 0
        
        if np.array(ini).size == p :
            ini_test = np.matmul(D, ini)
            lower_log = ini_test >= (lower + 1e-8)
            upper_log = ini_test <= (upper - 1e-8)
            bound_check = np.prod(np.multiply(lower_log,upper_log))
            
            if bound_check == 0:
                print('initial is outside or too close from boundary, will be auto-corrected by MP-inverse!')
        elif bound_check == 0:
            ini = np.matmul(np.linalg.pinv(D), (lower + upper)/2)
            
    if any(np.array([burn,thin,n]) % 1 != 0) :
        print('burn, thin and n must be integer!\n')
        return []
    if any(np.array([burn,thin,n-1]) % 1 != 0):
        print("burn, thin must be non-negative interger, n must be positive integer\n")
    
    a = lower - np.matmul(D, Mean)
    b = upper - np.matmul(D, Mean)
    Sigma_chol = np.linalg.cholesky(Sigma)
    R = np.matmul(D, Sigma_chol)
    
    x = np.linalg.solve(Sigma_chol, ini - Mean) # initial value for the transformed tmvt
    
    keep_t = np.zeros( ((thin+1)*n+burn, p) )
    
    for i in range((thin+1)*n+burn):
        u = np.random.chisquare(df=nu)
        denom = math.sqrt(u/nu)
        
        lw = a*denom ; up = b*denom
        z0 = x*denom
        
        z = np.matrix(rtmvn(n=1, Mean=np.zeros(p), Sigma=np.diag(np.ones(p)), \
                  D=R, lower=lw, upper=up, burn=0, ini=z0)).reshape((p,1))
        
        keep_t[i,:] = (np.matmul(Sigma_chol,z/denom) + Mean).reshape(1,p)

    final_ind = np.array(range((thin+1)*n))
    final_ind = final_ind[0:len(final_ind):(thin+1)] + thin + burn
    
    return keep_t[final_ind,:]    

 
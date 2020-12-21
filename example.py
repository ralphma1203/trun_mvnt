
from trun_mvnt import rtmvn, rtmvt

Mean = np.array([0,0])
Sigma = np.array([1, 0.5, 0.5, 1]).reshape((2,2))

D = np.array([1,0,0,1,1,-1]).reshape((3,2))
lower = np.array([-2,-1,-2])
upper = np.array([2,3,5])


#D = np.array([1,0,0,1]).reshape((2,2))
#lower = np.array([-2,-1])
#upper = np.array([2,3])

tmp = rtmvn(500, Mean, Sigma, D, lower, upper, 100, 1)


tmp = rtmvt(500, Mean, Sigma, 5, D, lower, upper, 100, 1)   
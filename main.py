import numpy as np
def beta_hat(x,y):
    global sum_x
    x_trans = np.transpose(x)
    sum_x = np.matmul(x_trans,x)
    sum_y = np.matmul(x_trans,y)
    beta = np.matmul(np.linalg.inv(sum_x),sum_y)
    return beta

x = np.array([[1,2,9,8],
            [1,3,4,4],
            [1,3,5,2],
            [1,6,8,5],
            [1,3,3,7]])
y = np.array([[3],[8],[3],[2],[4]])

beta = beta_hat(x,y)
y_hat = x@beta
#Residuals
e = y-y_hat
e_trans = np.transpose(e)
sum_e = e_trans@e
#Degree of Freedom
n = x.shape[0]
k = x.shape[1]-1
#Variance
sigma_hat = sum_e/(n-k-1)
sum_x_inv = np.linalg.inv(sum_x)
varcov_beta_hat = sigma_hat*sum_x_inv
ess = y_hat 
print(ess)
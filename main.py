import numpy as np
import math as m
def beta_hat(x,y):
    global sum_x
    global sum_y
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
z = sum(y)

beta = beta_hat(x,y)
y_hat = x@beta
#Residuals
e = y-y_hat
e_trans = np.transpose(e)
rss = e_trans@e
#Degree of Freedom
n = x.shape[0]
k = x.shape[1]-1
#Variance
sigma_hat = rss/(n-k-1)
sum_x_inv = np.linalg.inv(sum_x)
varcov_beta_hat = sigma_hat*sum_x_inv
beta_trans = np.transpose(beta)
ess = (beta_trans@sum_y)-((z**2)/n)
f_cal = (ess/k)/(rss/(n-k-1))
var_beta_1 = varcov_beta_hat[1,1]
var_beta_2 = varcov_beta_hat[2,2]
var_beta_3 = varcov_beta_hat[3,3]
t_beta_1_hat = (beta[1,0]/m.sqrt(var_beta_1))
t_beta_2_hat = (beta[2,0]/m.sqrt(var_beta_2))
t_beta_3_hat = (beta[3,0]/m.sqrt(var_beta_3))
r_square = ess/(ess+rss)

print('PRESENTATION')
print(f'Yi = {beta[0,0]:.2f} + {beta[1,0]:.2f}Xi1 + {beta[2,0]:.2f}Xi2 + {beta[3,0]:.2f}Xi3')
print(f'T-stat =     ({t_beta_1_hat:.2f})    ({t_beta_2_hat:.2f})   ({t_beta_3_hat:.2f})')
print(f'Observation = {n}')
print(f'R-Square = {r_square}')
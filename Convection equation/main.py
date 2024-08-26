
import numpy as np
import h5py
import matplotlib.pyplot as plt
import PIKL_periodic as pikl
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1001)

def centering(shift_t, shift_x, scale_t, scale_x, data_t, data_x, data_zt, data_zx):
  data_t = scale_t*data_t-shift_t
  data_x = scale_x*(data_x-shift_x)
  data_zt = scale_t*data_zt-shift_t
  data_zx = scale_x*(data_zx-shift_x)
  return data_t, data_x, data_zt, data_zx

beta = 20 

shift_t, shift_x, scale_t, scale_x = 0.5, 0.5, 1/2.01, 2
s=1
L=0.5
m= 20
domain = "square periodic in x"
default_type = torch.float64
torch.set_default_dtype(default_type)

dT = pikl.DifferentialOperator({(1, 0): 1})
dX = pikl.DifferentialOperator({(0, 1): 1})
PDE = torch.pi*dT + beta*dX

lambda_n, mu_n =  0, 10**5

n_t = 100
n_x = 100

err_n = []
for i in range(10):
  t,x = torch.rand(n_t)-0.5, 2*torch.rand(n_x)-1
  t,x = torch.meshgrid(t,x)
  t,x = t.flatten(), x.flatten()
  t0, x0, t_test, x_test = t[0:n_x], x[0:n_x], t[n_x:], x[n_x:]

  f_star = torch.sin(torch.pi*x-beta*t)
  initial_condition = torch.tensor(f_star[0:n_x],  dtype=default_type)
  ground_truth = torch.tensor(f_star[n_x:].flatten(), dtype=default_type)

  n =torch.tensor(n_x).to(device)

  regression_vect = pikl.RFF_fit(t0, x0, initial_condition, s, m, lambda_n, mu_n, L, domain, PDE, device)
  estimator = pikl.RFF_estimate(regression_vect, t_test, x_test, s, m, n, lambda_n, mu_n, L, domain, PDE, device)

  error_u = np.sqrt(torch.mean(torch.square(torch.abs(estimator - ground_truth))).item())/np.sqrt(torch.mean(torch.square(ground_truth)).item())
  err_n.append(error_u)

print('Relative L2 error_u: %e' % (np.mean(err_n)))
print('Standard deviation: %e' % (np.std(err_n)))

plt.imshow(f_star.view(-1,n_x))
plt.colorbar()
plt.savefig('fstar.pdf')
plt.imshow(torch.real(estimator.view(-1,n_x)).numpy())
plt.savefig('estimator.pdf')



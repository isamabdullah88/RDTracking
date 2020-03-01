import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# Example 1
"""
x_k = np.array([0., 0]).reshape(-1,1)
P_k = 10.*np.eye(2)

R = np.array([.5])
Q = np.array([[0.0075, 0.015], [0.015, 0.03]])

n = 2; m = 1
dt = 1.

def fx(x_k, dt):
	x_k_tmp = x_k.copy()
	xout = np.empty_like(x_k_tmp)
	xout[0] = x_k_tmp[1] * dt + x_k_tmp[0]
	xout[1] = x_k_tmp[1]
	return xout


def hx(x_k):
	# TODO: Define measurement mdoel
	x_k_tmp = x_k.copy()
	return x_k_tmp[:1]
"""


# Example 2
from data import parse_data
data, gdata = parse_data()
datax, datay = data
datax = [x+50 for x in datax]
datay = [y+50 for y in datay]

# plt.plot(datax, datay); plt.show()
# gdata = [(i,i) for i in range(num_data)]
# theta_range = 180
# thetas = np.linspace(0, theta_range, num=theta_range)
# datax = [100*np.cos(np.deg2rad(theta)) for theta in thetas]
# datay = [100*np.sin(np.deg2rad(theta)) for theta in thetas]

# datax = [i for i in range(100)]
# datay = [i for i in range(100)]

data_count = len(datax)

### MODEL 1 ###
# Define initial conditions
x_k1 = np.array([datax[0], datay[0], 0, 0.]).reshape(-1,1)
P_k1 = np.diag(np.array([2, 2, 2, 2]))

Q1 = 1e-1*np.diag(np.array([.5,.5,.3,.3]))
R1 = np.diag(np.array([2.25, 2.25]))

n1 = 4; m1 = 2; dt1 = 1.

def fx1(x_k, dt):
	"""
	State transition function. May be non-linear.
	"""
	x, y, vx, vy = x_k.copy()
	
	x_f = vx*dt + x
	y_f = vy*dt + y

	vx_f = vx
	vy_f = vy

	return np.array([x_f, y_f, vx_f, vy_f])


def hx1(x_k):
	"""
	State to Measurement function. May be non-linear.
	"""
	x, y, _, _ = x_k.copy()
	return np.array([x,y])




### MODEL 2 ###
# Define initial conditions
x_k2 = np.array([datax[0], datay[0], 0, 0., 0.]).reshape(-1,1)
P_k2 = np.diag(np.array([2, 2, 2, 2, 2]))

Q2 = 2e-1*np.diag(np.array([.5,.5,.3,.3, .3]))
R2 = np.diag(np.array([2.25, 2.25, 2.25, 2.25]))

n2 = 5; m2 = 4; dt2 = 1.

def fx2(x_k, dt):
	"""
	State transition function. May be non-linear.
	"""
	x, y, v, theta, w = x_k.copy()
	
	x_f = v*np.cos(theta)*dt + x
	y_f = v*np.sin(theta)*dt + y
	v_f = v
	
	theta_f = w*dt + theta
	w_f = w

	return np.array([x_f, y_f, v_f, theta_f, w_f])


def hx2(x_k):
	"""
	State to Measurement function. May be non-linear.
	"""
	x, y, v, theta, _ = x_k.copy()
	return np.array([x, y, v, theta])



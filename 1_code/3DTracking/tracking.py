
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:03f}".format(x)})
from numpy.linalg import inv
from scipy.linalg import cholesky

import matplotlib.pyplot as plt


### NOTES ###
# Currently, it can only support theta from 0 to 360 i.e. only a full rotation. If vehicle
# makes more than one complete rotation, it won't be supported by the tracker.


class Tracking(object):
	"""Tracking using Unscented Kalman Filter for non-linear models"""
	def __init__(self, dt, n, m, Q, R, fx, hx, x0=None, P0=None):
		super(Tracking, self).__init__()
		self._n = n # state dimensions
		self._m = m # measurement dimensions
		self._dt = dt # time delta

		if x0 is None:
			self.x_k = np.zeros((self._n, 1))
		else:
			self.x_k = x0.copy()
		self.x_k_f = self.x_k.copy()
		self.x_k_a = self.x_k.copy()

		if P0 is None:
			self.P_k = np.eye(self._n)
		else:
			self.P_k = P0.copy()
		self.P_k_f = self.P_k.copy()
		self.P_k_a = self.P_k.copy()

		# Noise matrices
		self.Q = Q.copy()
		self.R = R.copy()

		# Sigma points
		self.x_sigma_f = np.zeros((n, 2*n+1))

		self.z_k_f = np.zeros((self._m, 1))
		self.Cov_Z = np.eye(self._m)
		self.Cov_XZ = np.zeros((self._n, self._m))

		self.fx = fx
		self.hx = hx

	def _gen_sigma(self):
		"""
		Generates sigma points.
		"""
		x_k = self.x_k.copy()
		P_k = self.P_k.copy()
		n = self._n

		kappa_ = 1
		fact = (n + kappa_)

		S_k = cholesky(fact * P_k).T
		
		x_sigmas = np.zeros((n, 2*n+1))
		x_sigmas[:,0] = np.squeeze(x_k)
		for i in range(n):
			x_sigmas[:, i+1] = np.squeeze(np.subtract(x_k, -S_k[:,i].reshape(-1,1)))
			x_sigmas[:, i+n+1] = np.squeeze(np.subtract(x_k, S_k[:,i].reshape(-1,1)))
			
		return x_sigmas


	def time_update(self, x_sigma):
		"""
		Time update state with motion model.
		"""
		n = self._n
		x_sigma_f = x_sigma.copy()

		for i in range(2*n+1):
			x_sigma_f[:,i] = self.fx(x_sigma[:,i], self._dt)
		
		x_k_f, P_k_f = self.calc_avg(x_sigma_f)

		P_k_f += self.Q

		return x_k_f, P_k_f, x_sigma_f


	def meas_update(self, x_sigma_f):
		"""
		Updates according to the measurement model.
		"""
		n = self._n
		m = self._m

		z_sigma_f = np.zeros((m, 2*n+1))

		for i in range(2*n+1):
			z_sigma_f[:,i] = self.hx(x_sigma_f[:,i])

		z_k_f, Cov_Z = self.calc_avg(z_sigma_f)
		Cov_Z += self.R

		return z_k_f, Cov_Z, z_sigma_f

	def gen_weights(self):
		"""
		Generate weights for weighted average of sigma points.
		"""
		n = self._n
		Wm = [(1-0.33)/(2*n) for _ in range(2*n+1)]; Wm[0] = 0.33
		Wc = [(1-0.33)/(2*n) for _ in range(2*n+1)]; Wc[0] = 0.33

		return Wm, Wc

	def calc_avg(self, sigmas):
		"""
		Estimates the mean of sigma points using weights
		"""
		n = self._n
		
		Wm, Wc = self.gen_weights()

		x_k_avg = Wm[0]*sigmas[:,0].reshape(-1,1)

		for i in range(1, 2*n+1):
			x_k_avg = x_k_avg + (Wm[i]*sigmas[:,i].reshape(-1,1))

		resid0 = np.subtract(sigmas[:,0].reshape(-1,1), x_k_avg)
		P_k_avg = Wc[0] * np.outer(resid0, resid0)

		for i in range(1, 2*n+1):
			# TODO: Weighted update for cov_upd
			resid = np.subtract(sigmas[:,i].reshape(-1,1),x_k_avg)
			P_k_avg += Wc[i]*np.outer(resid, resid)

		return x_k_avg.reshape(-1,1), P_k_avg

	def calc_Cov_XZ(self, z_k_f, x_sigma_f, z_sigma_f):
		""" 
		Estimate cross covriance between state and measurement residual
		"""
		x_k_f = self.x_k_f.copy()
		n = self._n
		m = self._m
		_, Wc = self.gen_weights()

		Cov_XZ = np.zeros((n, m))
		
		for i in range(2*n+1):
			# TODO: Weighted update for cov_upd
			cov1 = x_sigma_f[:,i].reshape(-1,1) - x_k_f
			cov2 = (z_sigma_f[:,i].reshape(-1,1) - z_k_f)
			
			Cov_XZ += Wc[i]*np.outer(cov1, cov2)

		return Cov_XZ

	def propagate_state(self):
		"""
		Propagates state in time, updating state and covariance
		"""
		x_sigma = self._gen_sigma()
		
		self.x_k_f, self.P_k_f, x_sigma_f = self.time_update(x_sigma)

		self.x_sigma_f = x_sigma_f

	def propagate_meas(self):
		"""
		Propagates measurement in time, updating z and cov(z)
		"""
		x_sigma_f = self.x_sigma_f

		z_k_f, Cov_Z, z_sigma_f = self.meas_update(x_sigma_f)
		
		Cov_XZ = self.calc_Cov_XZ(z_k_f, x_sigma_f, z_sigma_f)
		
		self.z_k_f, self.Cov_Z, self.Cov_XZ = z_k_f, Cov_Z, Cov_XZ

	def assim_data(self, z_k):
		"""
		Performs data assimilation step. Computing kalman gain, and then estimating posterior.
		"""
		
		K_k = np.dot(self.Cov_XZ, inv(self.Cov_Z))
		
		
		y = z_k.reshape(-1,1) - self.z_k_f
		
		x_k_a = self.x_k_f + np.dot(K_k, y)
		P_k_a = self.P_k_f - np.dot(K_k, np.dot(self.Cov_Z, K_k.T))
		
		self.x_k_a = x_k_a
		self.P_k_a = P_k_a

		# Prepare for next step
		self.x_k = x_k_a
		self.P_k = P_k_a
		

def gen_sigma(x_k_, P_k, n, w=0.5):
	"""
	Generates sigma points.
	"""
	x_k = x_k_.copy()

	# alpha_ = 0.5
	kappa_ = 1
	# lambda_ = alpha_**2 * (n+kappa_) - n
	# fact = np.sqrt(n+lambda_)
	fact = (n + kappa_)

	S_k = cholesky(fact * P_k).T
	print('S_k: '); print(S_k)
	# # print('fact: ', fact)
	x_k_f = np.zeros((n, 2*n+1))
	x_k_f[:,0] = np.squeeze(x_k)
	for i in range(n):
		x_k_f[:, i+1] = np.squeeze(np.subtract(x_k, -S_k[:,i].reshape(-1,1)))
		x_k_f[:, i+n+1] = np.squeeze(np.subtract(x_k, S_k[:,i].reshape(-1,1)))
		
	# # print('x_k_f: '); # print(x_k_f)
	# plt.plot(x_k_f[0,:], x_k_f[1,:], 'g.'); plt.show()
	# plt.plot(x_k[0,0], x_k[1,0], 'r.')
	# print('sigmas: '); print(x_k_f)
	return x_k_f



def time_update(x_k, dt=1., thresh=1e2):
	"""
	Time update state with motion model.
	"""
	xout = np.empty_like(x_k)
	xout[0] = x_k[1] * dt + x_k[0]
	xout[1] = x_k[1]
	return xout
	# x_f = v*np.cos(theta)*T + x
	# y_f = v*np.cos(theta)*T + y
	# theta_f = w*T + theta
	# v_f = v
	# w_f = w
	# Linear Model to check
	# x_f = vx*T + x
	# y_f = vy*T + y
	# vx = vx
	# vy = vy

	# else:
	# 	x_f = v*np.cos(theta)*T + x
	# 	y_f = v*np.sin(theta)*T + y
	# 	theta_f = theta
	# 	v_f = v
	# 	w_f = w

	# x_k_f = np.array([x_f, y_f])

	# return x_k_f


def meas_update(x_k):
	"""
	Updates according to the measurement model.
	"""
	# TODO: Define measurement mdoel
	# print('x_k: ', x_k)
	return x_k[:1]



def calc_avg(x_sigma_f, n):
	"""
	Estimates the mean of sigma points using weights
	"""
	# alpha_ = .5
	# kappa_ = 3
	# lambda_ = alpha_**2 * (n+kappa_)
	# w0 = lambda_/(n+lambda_)
	# wi = 1/(2*(n+lambda_))
	
	Wm = [0.33333333, 0.16666667, 0.16666667, 0.16666667, 0.16666667]
	Wc = [0.33333333, 0.16666667, 0.16666667, 0.16666667, 0.16666667]

	x_k_avg = Wm[0]*x_sigma_f[:,0].reshape(-1,1)

	# # print('xkavg: ', x_k_avg.shape)
	# # print('n: ', 2*n+1)
	for i in range(1, 2*n+1):
		x_k_avg = x_k_avg + (Wm[i]*x_sigma_f[:,i].reshape(-1,1))
		# # print('x_k_avg sigma: '); # print(Wm[i]*x_sigma_f[:,i].reshape(-1,1))

	# beta_ = 2
	# w0 = lambda_/(n+lambda_) + 1 - alpha_**2 + beta_
	# w0 /= (2*n)

	# # print('lambda: ', lambda_)
	# # print('w0: ', w0)
	# # print('wi: ', wi)
	# s = x_sigma_f.shape[0]
	resid0 = np.subtract(x_sigma_f[:,0].reshape(-1,1), x_k_avg)
	P_k_avg = Wc[0] * np.outer(resid0, resid0)

	for i in range(1, 2*n+1):
		# TODO: Weighted update for cov_upd
		resid = np.subtract(x_sigma_f[:,i].reshape(-1,1),x_k_avg)
		# # print('resid: ', resid.shape)
		# # print('outer: ', np.outer(resid, resid).shape)
		P_k_avg += Wc[i]*np.outer(resid, resid)

	# # print('xkavg: ', x_k_avg.shape)
	# # print('P_k_avg: ', P_k_avg.shape)
	# exit()
	return x_k_avg.reshape(-1,1), P_k_avg
	
def calc_Cov_XZ(x_k_f, z_k_f, x_sigma_f, z_sigma_f, n, m):
	# Computing cross-cov
	# alpha_ = 0.5
	# kappa_ = 3-n
	# lambda_ = alpha_**2 * (n+kappa_) - n
	# w0 = lambda_/(n+lambda_)
	# wi = 1/(2*(n+lambda_))

	Wc = [0.33333333, 0.16666667, 0.16666667, 0.16666667, 0.16666667]

	Cov_XZ = np.zeros((n, m))
	
	for i in range(2*n+1):
		# TODO: Weighted update for cov_upd
		cov1 = x_sigma_f[:,i].reshape(-1,1) - x_k_f
		cov2 = (z_sigma_f[:,i].reshape(-1,1) - z_k_f)
		
		Cov_XZ += Wc[i]*np.outer(cov1, cov2)

	return Cov_XZ

def propagate_state(x_k, P_k, Q_k, W=None):
	"""
	Propagates state in time, updating state and covariance
	"""
	n = x_k.shape[0]
	x_sigma = gen_sigma(x_k, P_k, 2, w=0.5)
	x_sigma_f = x_sigma.copy()
	for i in range(2*n+1):
		x_sigma_f[:,i] = time_update(x_sigma[:,i])
	
	x_k_f, P_k_f = calc_avg(x_sigma_f, n)
	# # print('P_k_f 1: '); # print(P_k_f)
	# print('\nx_sigma_f: '); print(x_sigma_f)
	
	# Cov propagation
	P_k_f = P_k_f + Q_k
	
	return x_k_f, P_k_f, x_sigma, x_sigma_f

def propagate_meas(x_k, x_k_f, P_k, P_k_f, R_k, x_sigma, x_sigma_f):
	"""
	Propagates measurement in time, updating z and cov(z)
	"""
	n = 2
	m = 1
	# z_sigma = gen_sigma(x_k, P_k, 2)
	z_sigma_f = np.zeros((m, 2*n+1))
	print('\nx_sigma_f: '); print(x_sigma)
	for i in range(2*n+1):
		z_sigma_f[:,i] = meas_update(x_sigma_f[:,i])
	print('\nz_sigma_f: '); print(z_sigma_f)
	# TODO: Apply weighted mean
	z_k_f, Cov_Z = calc_avg(z_sigma_f, n)
	# print('zkf: '); # print(z_k_f)
	# exit()
	
	# # print('zkf: ', z_k_f.shape)
	# exit()
	Cov_Z = Cov_Z + R_k
	# # print('z_k_f: '); # print(z_k_f)
	# print('Cov_Z: '); # print(Cov_Z)
	# exit()
	Cov_XZ = calc_Cov_XZ(x_k_f, z_k_f, x_sigma_f, z_sigma_f, n, m)
	# print('Cov_XZ: '); # print(Cov_XZ)
	# exit()
		
	return z_k_f, Cov_Z, Cov_XZ


def assim_data(x_k_f, P_k_f, z_k, z_k_f, Cov_Z, Cov_XZ):
	"""
	Performs data assimilation step. Computing kalman gain, and then estimating posterior.
	"""
	# # print('Cov_XZ: '); # print(Cov_XZ)
	# # print('Cov_Z: '); # print(inv(Cov_Z))
	K_k = np.dot(Cov_XZ, inv(Cov_Z))
	print('K_k: '); print(K_k)
	# K_k = Cov_XZ @ inv(Cov_Z)

	# # print('x_k_f: '); # print(x_k_f)
	# # print('z_k: '); # print(z_k)
	# # print('z_k_f: '); # print(z_k_f)
	# # print('K_k: ', ); # print(K_k)
	# # print('K_k (mul): '); # print(K_k @ Cov_Z @ K_k.T)
	# print('zk: ', z_k)
	y = z_k - z_k_f
	# print('z_k_f: ', z_k_f.T)
	print('residual z: ', y)
	print('res dot: '); print(np.dot(K_k, y))
	x_k_a = x_k_f + np.dot(K_k, y)
	P_k = P_k_f - np.dot(K_k, np.dot(Cov_Z, K_k.T))
	# print('x_k_a: '); print(x_k_a)
	# # print('P_k1: '); # print(P_k)
	return x_k_a, P_k


"""
# from data import parse_data
# data, gdata = parse_data()
# datax, datay = data
num_data = 1000
gdata = [(i,i) for i in range(num_data)]
datax = [i for i in range(num_data)]
datay = [0 for i in range(num_data)]

data_count = len(datax)

# Define initial conditions
x_k = np.array([datax[0], datay[0], 0, 0]).reshape(-1,1)

P_k = np.diag(np.array([2, 2, 2, 2]))

# P_k = 1e-1 * P_k

Q_k = 1e3*np.diag(np.array([2.5,2.5,2.3,2.3]))
R_k = 1e-6*np.diag(np.array([2.25, 2.25]))


states = []
vars_pk = []

# plt.figure()
# plt.plot(datax, datay, 'r*')

for i in range(1, data_count, 1):
	x, y = (datax[i], datay[i])
	# theta = np.arctan((y-datay[i-1])/(x-datax[i-1]))
	z_k = np.array([x,y]).reshape(-1,1)

	# # print('P_k: '); # print(P_k)
	vars_pk.append(np.sum(np.abs(P_k)))
	# # print('z_k: '); # print(z_k)
	x_k_f, P_k_f, x_sigma, x_sigma_f = propagate_state(x_k, P_k, Q_k)
	# # print('x_k_f: '); # print(x_k_f[0:2])
	# # print('P_k_f: '); # print(np.sum(P_k_f))
	# x_k = x_k_f
	# P_k = P_k_f
	z_k_f, Cov_Z, Cov_XZ = propagate_meas(x_k, x_k_f, P_k, R_k, x_sigma, x_sigma_f)

	#2) Measurement Update.
	# # print('P_k_a: '); # print(P_k)
	x_k_a, P_k = assim_data(x_k_f, P_k_f, z_k, z_k_f, Cov_Z, Cov_XZ)
	# # print('theta: ', x_k_a[2])
	x_k = x_k_a
	

	states.append(x_k)

x = [s[0] for s in states]
y = [s[1] for s in states]
thetas = [s[2] for s in states]
plt.figure()
plt.plot(datax, datay, 'r.')
plt.plot(x, y, 'b')
plt.figure()
plt.plot(vars_pk)
plt.figure()
vs = [s[2] for s in states]
plt.plot(vs, 'y')
plt.show()
"""
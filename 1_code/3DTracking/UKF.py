
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:03f}".format(x)})
from numpy.linalg import inv
from scipy.linalg import cholesky

import matplotlib.pyplot as plt


### NOTES ###
# Currently, it can only support theta from 0 to 360 i.e. only a full rotation. If vehicle
# makes more than one complete rotation, it won't be supported by the tracker.


class UKF(object):
	"""Tracking using Unscented Kalman Filter for non-linear models"""
	def __init__(self, dt, n, m, Q, R, fx, hx, x0=None, P0=None):
		super(UKF, self).__init__()
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
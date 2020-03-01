import matplotlib.pyplot as plt

from numpy.random import randn
import numpy as np

from UKF import UKF
from models import dt1, n1, m1, Q1, R1, fx1, hx1, x_k1, P_k1
from models import dt2, n2, m2, Q2, R2, fx2, hx2, x_k2, P_k2



class Tracker(object):
	"""
	Combines Linear and Angular model to account for correct orientation estimation
	and trajectory projection estimation.
	"""
	def __init__(self):
		super(Tracker, self).__init__()

		self._tracker1 = UKF(dt1, n1, m1, Q1, R1, fx1, hx1, x0=x_k1, P0=P_k1)
		self._tracker2 = UKF(dt2, n2, m2, Q2, R2, fx2, hx2, x0=x_k2, P0=P_k2)

	def _get_theta(self, vx, vy):
		"""
		Returns normalized theta_f in radians given theta in radians.
		"""
		theta = np.rad2deg(np.arctan2(vy,vx))
		if theta < 0.: theta += 360
		return np.deg2rad(theta)

	def propagate(self):
		"""
		Propagates the state and measurement models of the hybrid filter.
		"""
		self._tracker1.propagate_state()
		self._tracker1.propagate_meas()

		self._tracker2.propagate_state()
		self._tracker2.propagate_meas()


	def update(self, z_k=None):
		"""
		Performs data assimilation step for the hybrid filter.
		z_k: Measurement data for update. np array of locations formatted: [x, y]. If None
		it will just propogate as trajectory projection estimation.
		"""
		if z_k is None:
			self._tracker1.x_k = self._tracker1.x_k_f
			self._tracker1.P_k = self._tracker1.P_k_f

			self._tracker2.x_k = self._tracker2.x_k_f
			self._tracker2.P_k = self._tracker2.P_k_f
		else:
			self._tracker1.assim_data(z_k)

			# Use some of the state elements for tracker2

			x_z, y_z, vx_z, vy_z = self._tracker1.x_k
			# Get resultant velocity from components
			v_z = np.sqrt(vx_z**2, vy_z**2)
			theta_z = self._get_theta(vx_z, vy_z)
			z_k2 = np.array([x_z, y_z, v_z, theta_z])

			self._tracker2.assim_data(z_k2)

	def get_state(self):
		"""
		Returns updated state elements as list. [x,y,vx,vy,v,theta]
		"""
		_, _, vx, vy = self._tracker1.x_k
		x, y, v, theta, _ = self._tracker2.x_k

		return [x, y, vx, vy, v, theta]

	def get_cov(self):
		"""
		Returns updated covariance matrix as (5X5) numpy array.
		"""
		return self._tracker2.P_k


def plot_states(states, datax, datay):

	xs = [s[0] for s in states]
	ys = [s[1] for s in states]
	vx = [s[2] for s in states]
	vy = [s[3] for s in states]
	v = [s[4] for s in states]
	thetas = [s[5] for s in states]
	
	plt.figure()
	plt.title('V')
	plt.plot(v)
	# plt.figure()
	# plt.plot(vx1)
	# plt.title('Vx1')
	# plt.figure()
	# plt.plot(vy1)
	# plt.title('Vy1')
	plt.figure()
	plt.plot(thetas)
	plt.title('Thetas')
	# plt.xlim(0, 1000)
	# plt.ylim(-1,10)
	plt.figure()
	plt.title('Track')
	plt.plot(datax, datay, marker='x', ls='', color='c')
	plt.plot(xs, ys, 'r')
	# plt.figure()
	# plt.plot(covs1)
	# plt.title('Covariances')
	plt.show()

def main():

	from models import datax, datay, data_count

	states = []
	tracker = Tracker()

	for i in range(data_count):

		if i%100==0: print('Processed: ', i)

		tracker.propagate()

		z_k = np.array([datax[i], datay[i]])
		if i < 1900:
			tracker.update(z_k=z_k)
		else:
			tracker.update()

		state = tracker.get_state()

		states.append(state)

	plot_states(states, datax, datay)


if __name__ == "__main__":
	main()
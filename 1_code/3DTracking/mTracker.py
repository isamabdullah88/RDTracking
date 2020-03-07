
import os

import cv2
import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from data import cuboid_data, get_cuboid_centers
from tracker import Tracker
from visTrackers import visualize

cuboids_list = cuboid_data()
centers_list = get_cuboid_centers(cuboids_list)

# TODO:
# 1) Enable unique id for each tracker.

class MTracker(object):
	"""
	Implements multi-object tracking.
	"""
	def __init__(self, boundx, boundy):
		super(MTracker, self).__init__()

		# x boundary
		self._xmin, self._xmax = boundx
		# y boundary
		self._ymin, self._ymax = boundy

		# Maintained list of trackers
		self._trackers = []

		# Assign each tracker a unique id
		self._tracker_ids = {}

	def init_trackers(self, init_meas):
		if self._tracker_ids:
			strt_idx = np.max(list(self._tracker_ids.keys()))+1
		else:
			strt_idx = 0
		for i,pt in enumerate(init_meas):
			x_k1 = np.array([pt[0], pt[1], 0., 0.]).reshape(-1,1)
			x_k2 = np.array([pt[0], pt[1], 0., 0.,0.]).reshape(-1,1)
			tracker = Tracker(x_k1, x_k2)

			self._trackers.append(tracker)
			print('id idx: ', strt_idx+i)
			self._tracker_ids[strt_idx+i] = strt_idx+i


	def propagate_trackers(self):
		for i in range(len(self._trackers)):
			self._trackers[i].propagate()

	def _construct_cost(self, meas):
		cost = np.zeros((len(self._trackers), len(meas)))

		for i, tracker in enumerate(self._trackers):
			for j,pt in enumerate(meas):
				sxy = np.squeeze(tracker.get_state()[:2])
				zxy = np.squeeze(np.array(pt[:2]))

				cost[i,j] = norm(sxy-zxy)

		# normalizing to (0,1)
		cost /= np.max(cost)
		return cost

	def update_trackers(self, meas):
		cost = self._construct_cost(meas)

		row_idx, col_idx = linear_sum_assignment(cost)

		for i in range(len(self._trackers)):
			if i < len(row_idx):
				r,c = (row_idx[i], col_idx[i])
				print('cost: (r,c)=>(cost) ', ((r,c), cost[r,c]))
				z_k = meas[c][:2]
				self._trackers[r].update(z_k=z_k)
			else:
				self._trackers[i].update()

		new_meas = []
		if len(meas) > len(self._trackers):
			for i in range(len(meas)):
				if not (i in col_idx):
					new_meas.append(meas[i])

			print('new meas: ', new_meas)	
			# Initialize new state for new measurement point
			self.init_trackers(new_meas)

		# Kicking out of range trackers
		for i,tracker in enumerate(self._trackers):
			x,y = tracker.get_state()[:2]

			num_tracker = len(self._trackers)
			if (x < self._xmin) or (x > self._xmax) or (y < self._ymin) or (y > self._ymax):
				# Pop out
				self._trackers.pop(i)


				del self._tracker_ids[i]
				
				# Adjusting indices of rest
				for k in self._tracker_ids.keys():
					if k > i:
						self._tracker_ids[k-1] = self._tracker_ids.pop(k)



data_count = len(centers_list)

xl, xr = (42.499606417905724, 42.499868)
yl, yr = (90.696186, 90.69658661844755)
mTracker = MTracker((xl, xr), (yl, yr))
mTracker.init_trackers(centers_list[0])


# fig = plt.figure(figsize=(12,10))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlim(xl, xr)
# ax.set_ylim(yl, yr)

vel = []

img_dir = '../../0_data/c035/images'

for i in range(50):
	meas = centers_list[i]

	mTracker.propagate_trackers()

	mTracker.update_trackers(meas)

	# for center in meas:
	# 	ax.plot(center[0], center[1], 'r*')

	vel.append(mTracker._trackers[0].get_state()[4])
	states = []
	for tracker in mTracker._trackers:
		state = tracker.get_state()
		states.append([state[0], state[1]])
		# ax.plot(state[0], state[1], 'g.')

	visualize(mTracker._trackers, mTracker._tracker_ids, show=False)
	# show image
	# img_path = os.path.join(img_dir, '{0:06d}.jpg'.format(i))
	# img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
	# plt.figure()
	# plt.imshow(img)
	out_plot_path = os.path.join('../../2_outputs/trackPlots', '{0:06d}'.format(i))
	plt.savefig(out_plot_path, dpi=100)
	# if i==26: 
	# 	for i,tracker in enumerate(mTracker._trackers):
	# 		print('state: ', tracker.get_state()[:2])
	# 	for i in range(len(meas)):
	# 		print('meas: ', meas[i])
	# 	plt.show()
	# 	exit()
	plt.close(); plt.cla()

	

# print('x lim: '); print(ax.get_xlim())
# print('y lim: '); print(ax.get_ylim())
# plt.figure()
# plt.plot(vel)
# plt.show()
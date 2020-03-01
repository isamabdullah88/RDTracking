import matplotlib.pyplot as plt

from numpy.random import randn
import numpy as np

from tracking import Tracking
from models import dt1, n1, m1, Q1, R1, fx1, hx1, x_k1, P_k1
from models import dt2, n2, m2, Q2, R2, fx2, hx2, x_k2, P_k2
from models import datax, datay, data_count

tracker1 = Tracking(dt1, n1, m1, Q1, R1, fx1, hx1, x0=x_k1, P0=P_k1)
tracker2 = Tracking(dt2, n2, m2, Q2, R2, fx2, hx2, x0=x_k2, P0=P_k2)

states1 = []
states2 = []
covs1 = []
covs2 = []

def get_theta(vx, vy):
    """
    Returns normalized theta_f in radians given theta in radians.
    """
    theta = np.rad2deg(np.arctan2(vy,vx))
    if theta < 0.: theta += 360
    return np.deg2rad(theta)


for i in range(data_count):
    
    # Tracker 1
    z_k1 = np.array([datax[i], datay[i]])
    tracker1.propagate_state()
    tracker1.propagate_meas()

    tracker2.propagate_state()
    tracker2.propagate_meas()

    print('Processed: ', i)

    if i < 1900:
        tracker1.assim_data(z_k1)
        
        x_z, y_z, vx_z, vy_z = tracker1.x_k
        v_z = np.sqrt(vx_z**2, vy_z**2)
        theta_z = get_theta(vx_z, vy_z)
        z_k2 = np.array([x_z, y_z, v_z, theta_z])
        tracker2.assim_data(z_k2)

    else:
        tracker1.x_k = tracker1.x_k_f
        tracker1.P_k = tracker1.P_k_f

        tracker2.x_k = tracker2.x_k_f
        tracker2.P_k = tracker2.P_k_f

    states1.append(tracker1.x_k)
    states2.append(tracker2.x_k)
    covs1.append(np.sum(np.abs(tracker1.P_k)))
    covs2.append(np.sum(np.abs(tracker2.P_k)))

xs1 = [s[0] for s in states1]
ys1 = [s[1] for s in states1]
vx1 = [s[2] for s in states1]
vy1 = [s[3] for s in states1]
v1 = [np.sqrt(x**2+y**2) for x,y in zip(vx1,vy1)]

thetas1 = []
for x,y in zip(vx1, vy1):
    theta = np.rad2deg(np.arctan2(y,x))
    if theta < 0.: theta += 360
    thetas1.append(theta)
    
xs2 = [s[0] for s in states2]
ys2 = [s[1] for s in states2]
v2 = [s[2] for s in states2]
thetas2 = [s[3] for s in states2]
    


plt.figure()
plt.title('V2')
plt.plot(v2)
# plt.figure()
# plt.plot(vx1)
# plt.title('Vx1')
# plt.figure()
# plt.plot(vy1)
# plt.title('Vy1')
plt.figure()
plt.plot(thetas2)
plt.title('Thetas2')
# plt.xlim(0, 1000)
# plt.ylim(-1,10)
plt.figure()
plt.title('Track2')
plt.plot(datax, datay, marker='x', ls='', color='c')
plt.plot(xs2, ys2, 'r')
# plt.figure()
# plt.plot(covs1)
# plt.title('Covariances')
plt.show()
import os

import matplotlib.pyplot as plt


def parse_data(plot=False):
    data_path = '../../0_data/obj_pose-laser-radar-synthetic-input.txt'

    gdatax = []
    gdatay = []

    datax = []
    datay = []

    with open(data_path, 'r') as f:
        data_tmp = f.read().split('\n')
        for d_tmp in data_tmp:
            d_tmp = d_tmp.split('\t')
            if d_tmp[0] == 'R':
                gdatax.append(float(d_tmp[5]))
                gdatay.append(float(d_tmp[6]))
            elif d_tmp[0] == 'L':
                datax.append(float(d_tmp[1]))
                datay.append(float(d_tmp[2]))

    if plot:
        plt.plot(datax, datay, 'r*')
        plt.plot(gdatax, gdatay, 'g')
        plt.show()

    return (datax, datay), (gdatax, gdatay)


data, gdata = parse_data()
datax, datay = data
datax = [x+50 for x in datax]
datay = [y+50 for y in datay]

data_count = len(datax)



# plt.plot(datax, datay); plt.show()
# gdata = [(i,i) for i in range(num_data)]
# theta_range = 180
# thetas = np.linspace(0, theta_range, num=theta_range)
# datax = [100*np.cos(np.deg2rad(theta)) for theta in thetas]
# datay = [100*np.sin(np.deg2rad(theta)) for theta in thetas]

# datax = [i for i in range(100)]
# datay = [i for i in range(100)]
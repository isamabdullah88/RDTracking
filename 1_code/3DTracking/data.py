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
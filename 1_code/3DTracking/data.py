import os
from glob import glob

import numpy as np
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

def udacity_data():
    data, gdata = parse_data()
    datax, datay = data
    datax = [x+50 for x in datax]
    datay = [y+50 for y in datay]

    data_count = len(datax)

    return datax, datay, data_count

def cuboid_data():
    data_dir = '../../0_data/Cuboids'
    data_path_list = glob(os.path.join(data_dir, '*.txt'))
    data_path_list.sort()

    def parse(data_path):
        data = []
        with open(data_path, 'r') as f:
            data = f.read().split('\n')
        
        cuboids = []
        for d in data:
            d = d.split(',')
            cuboid = []
            for pt in d:
                pt = pt.split(' ')
                if pt[0] == '':
                    continue
                pt_tmp = np.array([float(x) for x in pt])
                cuboid.append(pt_tmp)

            if cuboid:
                cuboids.append(cuboid)
        
        return cuboids

    cuboids_list = []
    for data_path in data_path_list:
        cuboids = parse(data_path)
        
        cuboids_list.append(cuboids)

    return cuboids_list

def get_center(cuboid):
    pt1, pt2, pt3, pt4 = cuboid

    ptd1 = np.cross(pt1, pt3)
    ptd2 = np.cross(pt2, pt4)

    center = np.cross(ptd1, ptd2)
    center /= center[2]

    # In order to work for tracker
    # center *= 1e13

    return center[0:2]

def plot_cuboids(centers_list):
    for i in range(15):
        centers = centers_list[i]
        for center in centers:
            x = center[0]; y = center[1]
            plt.plot(x,y, '.')
    plt.show()

def normalize_data(centers_list, mean_c):
    for i,centers in enumerate(centers_list):
        for j,center in enumerate(centers):
            # center[0] -= mean_c[0]
            center[1] *= -1
            centers[j] = center
        centers_list[i] = centers

    return centers_list

def get_cuboid_centers(cuboids_list):
    mean_c = []
    centers_list = []
    for cuboids in cuboids_list:
        centers = []
        for cuboid in cuboids:
            center = get_center(cuboid)
            centers.append(center)
            mean_c.append([center[0], center[1]])
        centers_list.append(centers)
    
    mean_c = np.array(mean_c)
    mean_c = np.mean(mean_c, axis=0)
    
    centers_list = normalize_data(centers_list, mean_c)

    return centers_list

def main():
    cuboids_list = cuboid_data()
    centers_list = get_cuboid_centers(cuboids_list)
    print('len: ', len(centers_list))
    plot_cuboids(centers_list)

if __name__ == "__main__":
    main()

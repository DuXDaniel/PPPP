import numpy as np
from numba import jit, prange
import scipy.integrate as spi
import scipy.interpolate as spip
import scipy.special as sps
import math as math
import matplotlib
import matplotlib.pyplot as plt
import time
import tqdm
import sys
import json
from cython.parallel import prange
from matplotlib.figure import Figure

@jit(nopython = True)
def meshgrid_custom(x,y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(x.size):
        for k in range(y.size):
            xx[j,k] = x[j]  # change to x[k] if indexing xy
            yy[j,k] = y[k]  # change to y[j] if indexing xy
    return xx, yy

@jit(nopython = True)
def dist_calc(vec):
    return np.sqrt(sum(vec*vec))

def spot_path_generator(distance): # [spots, paths]
    print('Generating initial spot list')

    dist_lim = distance # momenta units

    momenta_shifts = [[2,0,0],[0,2,0],[-1,1,0],[1,1,0]]

    x_shift = np.arange(-dist_lim,dist_lim)
    z_shift = np.arange(-dist_lim,dist_lim)

    [x_points,z_points] = meshgrid_custom(x_shift,z_shift)

    spot_list = np.zeros((len(x_points.flatten()),3))

    spot_list[:,0] = x_points.flatten()
    spot_list[:,1] = z_points.flatten()
    spot_list[:,2] = np.zeros(len(x_points.flatten()))

    print('Culling duplicates and out of bounds spots')

    val = 0
    while (val <= len(spot_list)-2):
        point_interest = spot_list[val,:]
        count = val+1
        while (count <= len(spot_list)-1):
            if (point_interest == spot_list[count,:]).all() or (dist_calc(spot_list[count,:]) >= dist_lim) or (sum(spot_list[count,:]) % 2 != 0):
                spot_list = np.delete(spot_list,count,0)
            else:
                count = count + 1
        
        if (dist_calc(point_interest) >= dist_lim) or (sum(point_interest) % 2 != 0):
            spot_list = np.delete(spot_list,val,0)
        else:
            val = val + 1

    print('Generating path lists for each spot')

    path_list = []

    for i in np.arange(spot_list.shape[0]):
        path_list.append([])
        for l in np.arange(-dist_lim,dist_lim):
            for p in np.arange(-dist_lim,dist_lim):
                for n in np.arange(-dist_lim,dist_lim):
                    for m in np.arange(-dist_lim,dist_lim):
                        cur_point = l*np.array(momenta_shifts[0]) + p*np.array(momenta_shifts[1]) + n*np.array(momenta_shifts[2]) + m*np.array(momenta_shifts[3])
                        if (dist_calc(np.array([l,p,n,m])) <= dist_lim):
                            if (cur_point == spot_list[i,:]).all():
                                cur_arr = path_list[i]
                                cur_arr.append([l,p,n,m])
                                path_list[i] = cur_arr

    spots = spot_list
    paths = path_list

    max_num_in_list = 0
    for i in np.arange(len(paths)):
        if len(paths[i]) > max_num_in_list:
            max_num_in_list = len(paths[i])
    
    paths_arr = np.zeros((spots.shape[0],max_num_in_list,4))
    paths_arr[:,:,:] = 1e99

    for i in np.arange(len(paths)):
        cur_paths = paths[i]
        for j in np.arange(len(cur_paths)):
            paths_arr[i,j,:] = cur_paths[j]

    return spots, paths_arr

def main(argv):
    for i in np.arange(14,25):
        full_init = time.time()
        data = {}
        [spots,paths] = spot_path_generator(i)
        data_dump = []
        data['spots'] = spots.tolist()
        data['paths'] = paths.tolist()
        data_dump.append(data)
        filename = "spot_" + str(i) + ".json"
        with open(filename, "w") as outfile:
            json_data = json.dump(data,outfile)
        full_elapsed = time.time() - full_init
        print(full_elapsed)

if __name__ == '__main__':
    main(sys.argv)
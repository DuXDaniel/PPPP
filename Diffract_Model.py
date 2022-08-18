import numpy as np
from numba import jit, prange
import scipy.integrate as spi
import scipy.interpolate as spip
import scipy.special as sps
import math as math
import time
import tqdm
import sys
import json
import multiprocessing as mp
from functools import partial

planck = 6.626e-34 #J.*s
c = 2.9979e8 # m./s
varepsilon = 8.8541878128e-12 # F/m, s^4*A**2/kg/m^3
hbar = planck/2/math.pi
mass_e = 9.11e-31 # kg
e = 1.602e-19 # Coulombs

@jit(nopython = True)
def temporal_gauss(z,t,sig_las):
    val = np.exp(-(2*(z-c*t)**2)/(sig_las**2*c**2))
    return val

@jit(nopython = True)
def omeg_las_sq(z, beam_waist, lam):
    z0 = np.pi*beam_waist**2/lam # Rayleigh range, nm
    val = beam_waist**2*(1+z**2/z0**2)
    return val

@jit(nopython = True)
def spatial_gauss(rho_xy,z,t, beam_waist,sig_las, lam):
    val = 1/np.pi/omeg_las_sq(z,beam_waist,lam)*np.exp(-(2*rho_xy**2)/(omeg_las_sq(z,beam_waist,lam)/temporal_gauss(z,t,sig_las)))
    return val

@jit(nopython = True)
def laser(rho_xy,z,t, beam_waist,sig_las, lam):
    val = spatial_gauss(rho_xy,z,t,beam_waist,sig_las,lam)*temporal_gauss(z,t,sig_las)
    return val

@jit(nopython = True)
def norm_laser_integrand(rho_xy,z,t,beam_waist,sig_las, lam):
    val = 2*np.pi*rho_xy*laser(rho_xy,z,t,beam_waist,sig_las,lam)
    return val

def laser_sum(t, gauss_limit, sig_las, beam_waist, lam):
    val = spi.dblquad(norm_laser_integrand, c*(t-gauss_limit*sig_las), c*(t + gauss_limit*sig_las), 0, gauss_limit*np.sqrt(omeg_las_sq(c*t, beam_waist, lam)), args=[t, beam_waist, sig_las, lam])
    return val

def Norm_Laser_Calculator(t_range,gauss_limit,sig_las,lam,w0,E_pulse): # [norm_func, norm_factor_array]
    freq = c/lam # Hz

    laser_sum_array = np.zeros(len(t_range))
    for i in np.arange(len(t_range)):
        val = laser_sum(t_range[i], gauss_limit, sig_las, w0, lam)
        laser_sum_array[i] = val[0]

    select_zero_laser_sum_array = np.where(laser_sum_array == 0, 1, 0)
    norm_factor_array = np.sqrt(E_pulse/(select_zero_laser_sum_array*1e308 + laser_sum_array)/varepsilon)
    norm_factor_array = norm_factor_array.astype(complex)
    norm_factor_array = norm_factor_array*(-1)*1j/freq
    return norm_factor_array

@jit(nopython = True)
def Electron_Generator(num,xoverArr,t_lim,vel,t_step): # [init_vels, timeArr]
    xoverArr = np.linspace(-xoverArr/2,xoverArr/2,round(np.sqrt(num)))
    timeArr = np.arange(-t_lim/2,t_lim/2,t_step)
    angularArray = np.linspace(0,2*math.pi,round(np.sqrt(num)))

    start_height = 1; # arbitrary example height

    test_time = start_height/vel
    x_pos = np.sin(xoverArr)*start_height*np.cos(angularArray)
    x_vel = -x_pos/test_time
    y_pos = np.sin(xoverArr)*start_height*np.sin(angularArray)
    y_vel = -y_pos/test_time

    [x_vel_grid,y_vel_grid] = meshgrid_custom(x_vel,y_vel)

    init_vels = np.zeros((len(x_vel_grid.flatten()),3))

    init_vels[:,0] = x_vel_grid.flatten()
    init_vels[:,1] = y_vel_grid.flatten()
    init_vels[:,2] = -vel*np.ones(len(x_vel_grid.flatten()))

    return init_vels, timeArr

@jit(nopython = True)
def meshgrid_custom(x,y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(x.size):
        for k in range(y.size):
            xx[j,k] = x[j]  # change to x[k] if indexing xy
            yy[j,k] = y[k]  # change to y[j] if indexing xy
    return xx, yy

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

@jit(nopython = True)
def dist_calc(vec):
    return np.sqrt(sum(vec*vec))

def besselmap(path,W_x,W_z,firstDiag,secondDiag,k):
    if sum(path[k,:])<=1e98:
        return (sps.jv(path[k,0],W_x)*sps.jv(path[k,1],W_z)*sps.jv(path[k,2], firstDiag)*sps.jv(path[k,3], secondDiag))**2
    else:
        return np.zeros(W_x.size)

def model_caller(prob_vals, spots, paths, W_x, W_z, firstDiag, secondDiag):
    path = paths.reshape(-1,paths.shape[2])

    totbess_matr = np.zeros((path.shape[0],W_x.size))
    k_arr = np.arange(path.shape[0])
    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        totbess_matr = pool.map(partial(besselmap, path,W_x,W_z,firstDiag,secondDiag),k_arr)

    totbess_matr = np.array(totbess_matr).reshape((paths.shape[0],paths.shape[1],W_x.size))

    for j in np.arange(paths.shape[0]):
        '''
        path = paths[j,:,:]
        totbess_matr = np.zeros((path.shape[0],W_x.size))
        for k in np.arange(path.shape[0]):
            if sum(path[k,:])<=1e98:
                totbess_matr[k,:] = (sps.jv(path[k,0],W_x)*sps.jv(path[k,1],W_z)*sps.jv(path[k,2], firstDiag)*sps.jv(path[k,3], secondDiag))**2
        '''
        prob_vals[j,:] = sum(totbess_matr[j,:,:],0)

    return prob_vals

### GENERATE FUNCTION THAT SAVES PRESET SPOT LISTS

def PPPP_calculator(e_res=350e-15,laser_res=350e-15,w0=100e-6,E_pulse=5e-6,beam_waist=100e-6,gauss_limit=4,ebeam_dxover=0,las_wav=517e-9,ebeam_vel=8.15e7,pos_adj_x=0,pos_adj_y=0,pos_adj_z=0):
    print('Seeding workspace with relevant information.')

    num_electron_MC_trial = int(4) # number of electrons to test per trial, must be square rootable
    num_electrons_per_stage = 100 # 1e6; must be square rootable

    theta =  0

    lam = las_wav # m
    w0 = beam_waist # m
    sig_las = laser_res # s
    z0 = math.pi*w0**2/lam # Rayleigh range, nm

    xover_slope = ebeam_dxover/300e-3 # 3 mm in 300 mm of travel
    xover_angle = math.atan(xover_slope) # degrees
    vel = ebeam_vel # velocity of electron, 200 kV, m./s
    gamma = lambda v: (1-(v/c)**2)**(-1/2)
    sig_ebeam = e_res # time resolution of ebeam, ps
    omeg_ebeam = lambda y: abs(xover_slope*y) # m
    e_wav = lambda v: hbar*2*math.pi/mass_e/v # v what units?????
    e_energ = lambda v: mass_e*((1/sqrt(1 - (v**2/c**2))) - 1) # v what units????

    ebeam_xover_size = 100e-6 # % m
    segments = 10
    x_range = np.linspace(-ebeam_xover_size,ebeam_xover_size,2*segments)
    z_range = np.linspace(-ebeam_xover_size,ebeam_xover_size,2*segments)

    t_mag = 15
    t_step = 10.**(-1*t_mag)#1e-15 # fs steps
    t_range = np.arange(-gauss_limit*sig_las,gauss_limit*sig_las,t_step)
    #norm_factor_array = Norm_Laser_Calculator(t_range,gauss_limit,sig_las,lam,w0,E_pulse)
    
    filename = "norm_factor_" + str(t_mag) + ".json"
    data = open(filename)
    data_dump = json.load(data)
    norm_factor_array = 1j*np.array(data_dump['norm_factor'],dtype=complex)

    [vels, elecTime] = Electron_Generator(num_electron_MC_trial,xover_angle,0,vel,t_step)

    pos_adj = np.ones(vels.shape)
    pos_adj[:,0] = pos_adj[:,0] + np.ones(vels.shape[0])*pos_adj_x
    pos_adj[:,1] = pos_adj[:,1] + np.ones(vels.shape[0])*pos_adj_z
    pos_adj[:,2] = pos_adj[:,2] + np.ones(vels.shape[0])*pos_adj_y

    dist_traveled = np.zeros((num_electron_MC_trial,3))
    phase_arr = np.zeros(num_electron_MC_trial)
    pos = vels*(t_range[1]) + pos_adj
    vel_arr = vels
    vel_mag_arr = np.sqrt(np.sum(vels*vels,1))
    moment_arr = gamma(vel_arr)*mass_e*vels
    moment_mag_arr = np.sqrt(np.sum(moment_arr*moment_arr,1))
    wavelength_arr = planck/moment_mag_arr
    energ_arr = np.sqrt(moment_mag_arr**2*c**2 + mass_e**2*c**4)

    init_wav = wavelength_arr
    init_vel_arr = moment_arr/np.transpose(np.ones((3,1))*energ_arr)*c*c
    init_vel_mag_arr = np.sqrt(np.sum(init_vel_arr*init_vel_arr,1))
    phase_exp = t_step*init_vel_mag_arr/init_wav*2*math.pi

    #[spots,paths] = spot_path_generator(10) # distance of 100 momenta units away from direct beam
    
    filename = "spot_" + str(10) + ".json"
    data = open(filename)
    data_dump = json.load(data)
    spots = np.array(data_dump['spots'])
    paths = np.array(data_dump['paths'])

    print('Starting calculation')

    pbar = tqdm.tqdm(total=t_range.size,desc='Current Calculation Progress',position=1,leave=True)
    
    for i in np.arange(t_range.size):
        random_sel = np.random.rand(num_electron_MC_trial)
    
        G_x = sig_las*np.sqrt(np.pi/2)*np.exp(-(2*pos[:,0])**2/(sig_las**2*c**2))*(sps.erf(t_range[i]*np.sqrt(2)/sig_las) - sps.erf((t_range[i]+t_step)*np.sqrt(2)/sig_las))
        omeg_las_sq_x = w0**2*(1+pos[:,0]**2/z0**2)
        rho_zy_sq = pos[:,1]**2 + pos[:,2]**2
        W_x = e**2/2/hbar/mass_e*norm_factor_array[i]**2*w0**2/omeg_las_sq_x*np.exp(-rho_zy_sq/omeg_las_sq_x)*G_x
        
        G_z = sig_las*np.sqrt(np.pi/2)*np.exp(-(2*pos[:,1])**2/(sig_las**2*c**2))*(sps.erf(t_range[i]*np.sqrt(2)/sig_las) - sps.erf((t_range[i]+t_step)*np.sqrt(2)/sig_las))
        omeg_las_sq_z = w0**2*(1+pos[:,1]**2/z0**2)
        rho_xy_sq = pos[:,0]**2 + pos[:,2]**2
        W_z = e**2/2/hbar/mass_e*norm_factor_array[i]**2*w0**2/omeg_las_sq_z*np.exp(-rho_xy_sq/omeg_las_sq_z)*G_z
        
        C0 = -e**2/2/hbar/mass_e*2*norm_factor_array[i]**2*w0**2/np.sqrt(omeg_las_sq_x)/np.sqrt(omeg_las_sq_z)*np.exp(-rho_xy_sq/omeg_las_sq_z - rho_zy_sq/omeg_las_sq_x)
        Tzx = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]-pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]+pos[:,1]-2*c*t_range[i])/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]+pos[:,1]-2*c*(t_range[i]+t_step))/(sig_las*c*np.sqrt(2))))
        Tznx = -sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]+pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]-pos[:,1]+2*c*t_range[i])/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]-pos[:,1]+2*c*(t_range[i]+t_step))/(sig_las*c*np.sqrt(2))))
        Tnzx = -sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]+pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((-pos[:,0]+pos[:,1]+2*c*t_range[i])/(sig_las*c*np.sqrt(2)))-sps.erf((-pos[:,0]+pos[:,1]+2*c*(t_range[i]+t_step))/(sig_las*c*np.sqrt(2))))
        Tnznx = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]-pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]+pos[:,1]+2*c*t_range[i])/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]+pos[:,1]+2*c*(t_range[i]+t_step))/(sig_las*c*np.sqrt(2))))

        prob_vals = np.zeros((spots.shape[0],num_electron_MC_trial))

        W_x = abs(W_x).real
        W_z = abs(W_z).real
        firstDiag = abs(C0*(Tzx+Tnznx)).real
        secondDiag = abs(C0*(Tznx+Tnzx)).real

        prob_vals = model_caller(prob_vals, spots, paths, W_x, W_z, firstDiag, secondDiag)
    
        prob_val_cumul = np.zeros(prob_vals.shape)
        indices = np.zeros(num_electron_MC_trial, dtype=int)
        
        for j in np.arange(num_electron_MC_trial):
            prob_val_cumul[:,j] = np.cumsum(prob_vals[:,j]/sum(prob_vals[:,j]))
            indices[j] = np.nonzero(prob_val_cumul > np.full(prob_val_cumul.shape,random_sel[j]))[0][0]
        
        print(prob_val_cumul)
        
        momenta_shift = spots[indices,:]*hbar/lam

        moment_arr = moment_arr + momenta_shift
        moment_mag_arr = np.sqrt(np.sum(moment_arr*moment_arr,1))
        energ_arr = np.sqrt(moment_mag_arr**2*c**2 + mass_e**2*c**4)
        vel_arr = moment_arr/np.transpose(np.ones((3,1))*energ_arr)*c*c
        
        dist_travel_step = t_step*vel_arr
        dist_travel_mag = np.sqrt(np.sum(dist_travel_step*dist_travel_step,1))
        pos = pos + dist_travel_step
        dist_traveled = dist_traveled + dist_travel_step
        wavelength_arr = planck/moment_mag_arr
        phase_step = dist_travel_mag/wavelength_arr*2*math.pi
        phase_arr = phase_arr + phase_step - phase_exp
        pbar.update(1)

    pbar.close()
    print(phase_arr)

    data = voxel_grid_cumulative_phase_data
    return data

def main(argv):
    PPPP_calculator()
    #sys.exit(appctxt.app.exec())

if __name__ == '__main__':
    main(sys.argv)

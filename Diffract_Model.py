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
    val = np.exp(-((z-c*t)**2)/(sig_las**2*c**2))
    return val

@jit(nopython = True)
def omeg_las_sq(z, beam_waist, lam):
    z0 = np.pi*beam_waist**2/lam # Rayleigh range, nm
    val = beam_waist**2*(1+z**2/z0**2)
    return val

@jit(nopython = True)
def spatial_gauss(rho_xy,z,t, beam_waist,sig_las, lam):
    val = beam_waist/np.sqrt(omeg_las_sq(z,beam_waist,lam))*np.exp(-(rho_xy**2)/(omeg_las_sq(z,beam_waist,lam)))
    return val

@jit(nopython = True)
def laser(rho_xy,z,t, beam_waist,sig_las, lam):
    val = spatial_gauss(rho_xy,z,t,beam_waist,sig_las,lam)*temporal_gauss(z,t,sig_las)
    return val

@jit(nopython = True)
def norm_laser_integrand(rho_xy,z,t,beam_waist,sig_las, lam):
    val = 2*np.pi*rho_xy*varepsilon*laser(rho_xy,z,t,beam_waist,sig_las,lam)**2
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
    norm_factor_array = np.sqrt(E_pulse/(select_zero_laser_sum_array*1e308 + laser_sum_array))
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

def dbl_KD_besselmap(path,W_x,W_z,firstDiag,secondDiag, dWx, dWz, dFirstDiag, dSecondDiag,k):
    if sum(path[k,:])<=1e98:
        first_part = dWx*(sps.jv(path[k,0],W_x)*(sps.jv(path[k,0]-1,W_x) - sps.jv(path[k,0]+1,W_x)))*sps.jv(path[k,1],W_z)**2*sps.jv(path[k,2], firstDiag)**2*sps.jv(path[k,3], secondDiag)**2
        second_part = dWz*(sps.jv(path[k,1],W_z)*(sps.jv(path[k,1]-1,W_z) - sps.jv(path[k,1]+1,W_z)))*sps.jv(path[k,0],W_x)**2*sps.jv(path[k,2], firstDiag)**2*sps.jv(path[k,3], secondDiag)**2
        third_part = dFirstDiag*(sps.jv(path[k,2],firstDiag)*(sps.jv(path[k,2]-1,firstDiag) - sps.jv(path[k,2]+1,firstDiag)))*sps.jv(path[k,1],W_z)**2*sps.jv(path[k,0], W_x)**2*sps.jv(path[k,3], secondDiag)**2
        fourth_part = dSecondDiag*(sps.jv(path[k,3],secondDiag)*(sps.jv(path[k,3]-1,secondDiag) - sps.jv(path[k,3]+1,secondDiag)))*sps.jv(path[k,1],W_z)**2*sps.jv(path[k,0], W_x)**2*sps.jv(path[k,2], firstDiag)**2
        return first_part + second_part + third_part + fourth_part
    else:
        return np.zeros(W_x.size)

def crs_KD_model_caller(prob_vals, spots, paths, W):
    path = paths

    totbess_matr = np.zeros((path.shape[0],W.size))
    
    for k in np.arange(path.shape[0]):
        totbess_matr[k,:] = sps.jv(path[k,1],W)*(sps.jv(path[k,1]-1,W)-sps.jv(path[k,1]+1,W)) #sps.jv(path[k,1],W)**2
    
    prob_vals = totbess_matr

    return prob_vals

def sgl_KD_model_caller(prob_vals, spots, paths, W):
    path = paths

    totbess_matr = np.zeros((path.shape[0],W.size))
    
    for k in np.arange(path.shape[0]):
        totbess_matr[k,:] = sps.jv(path[k,1]/2,W)*(sps.jv(path[k,1]/2-1,W)-sps.jv(path[k,1]/2+1,W)) #sps.jv(path[k,1]/2,W)**2
    
    prob_vals = totbess_matr

    return prob_vals

def dbl_KD_model_caller(prob_vals, spots, paths, W_x, W_z, firstDiag, secondDiag, dWx, dWz, dFirstDiag, dSecondDiag):
    '''
    path = paths.reshape(-1,paths.shape[2])

    totbess_matr = np.zeros((path.shape[0],W_x.size))
    k_arr = np.arange(path.shape[0])
    with mp.Pool(processes = mp.cpu_count()-1) as pool:
        totbess_matr = pool.map(partial(dbl_KD_besselmap, path,W_x,W_z,firstDiag,secondDiag),k_arr)

    totbess_matr = np.array(totbess_matr).reshape((paths.shape[0],paths.shape[1],W_x.size))
    '''
    for j in np.arange(paths.shape[0]):
        #'''
        path = paths[j,:,:]
        totbess_matr = np.zeros((path.shape[0],W_x.size))
        for k in np.arange(path.shape[0]):
            totbess_matr[k,:] = dbl_KD_besselmap(path,W_x,W_z,firstDiag,secondDiag, dWx, dWz, dFirstDiag, dSecondDiag, k)
        #'''
        prob_vals[j,:] = sum(totbess_matr,0)**2 # [j,:,:]

    return prob_vals

### GENERATE FUNCTION THAT SAVES PRESET SPOT LISTS

def PPPP_calculator(e_res=1e-12,laser_res=1e-12,E_pulse=5e-6,beam_waist=100e-6,gauss_limit=4,ebeam_dxover=0,las_wav=517e-9,ebeam_vel=2.0844e8,pos_adj_x=0,pos_adj_y=0,pos_adj_z=0,calcType=0,t_mag=15):
    print('Seeding workspace with relevant information.')

    num_electron_MC_trial = int(100) # number of electrons to test per trial, must be square rootable
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

    t_step = 10.**(-1*t_mag)#1e-15 # fs steps
    norm_factor_array = Norm_Laser_Calculator(np.array([0]),gauss_limit,sig_las,lam,w0,E_pulse)

    curtime = -2*sig_las
    timestop = 2*sig_las

    [vels, elecTime] = Electron_Generator(num_electron_MC_trial,xover_angle,0,vel,t_step)

    pos_adj = np.zeros(vels.shape)
    pos_adj[:,0] = np.ones(vels.shape[0])*pos_adj_x
    pos_adj[:,1] = np.ones(vels.shape[0])*pos_adj_z
    pos_adj[:,2] = np.ones(vels.shape[0])*pos_adj_y

    dist_traveled = np.zeros((num_electron_MC_trial,3))
    phase_arr = np.zeros(num_electron_MC_trial)
    pos = vels*(curtime) + pos_adj
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
    
    point_distance = 5
    if (calcType == 0):
        filename = "./crs_KD_spots/spot_" + str(point_distance) + ".json"
    elif (calcType == 1):
        filename = "./sgl_KD_spots/spot_" + str(point_distance) + ".json"
    elif (calcType == 2):
        filename = "./dbl_KD_spots/spot_" + str(point_distance) + ".json"
    data = open(filename)
    data_dump = json.load(data)
    spots = np.array(data_dump['spots'])
    paths = np.array(data_dump['paths'])

    print('Starting calculation')

    pbar = tqdm.tqdm(total=int((timestop-curtime)/t_step + 1),desc='Current Calculation Progress',position=1,leave=True)
    
    while curtime <= timestop:
        random_sel = np.random.rand(num_electron_MC_trial)

        if (calcType == 0):
            G = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0] - pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0] + pos[:,1] - 2*c*curtime)/(sig_las*c*np.sqrt(2))) - sps.erf((pos[:,0] + pos[:,1] - 2*c*(curtime + t_step))/(sig_las*c*np.sqrt(2))))
            omeg_las_sq_x = w0**2*(1+pos[:,0]**2/z0**2)
            rho_zy_sq = pos[:,1]**2 + pos[:,2]**2
            omeg_las_sq_z = w0**2*(1+pos[:,1]**2/z0**2)
            rho_xy_sq = pos[:,0]**2 + pos[:,2]**2
            W = -e**2/2/hbar/mass_e*2*norm_factor_array**2*w0**2/np.sqrt(omeg_las_sq_x)/np.sqrt(omeg_las_sq_z)*np.exp(-rho_xy_sq/omeg_las_sq_z - rho_zy_sq/omeg_las_sq_x)*G
            W = abs(W).real

            prob_vals = np.zeros((spots.shape[0],num_electron_MC_trial))

            prob_vals = crs_KD_model_caller(prob_vals, spots, paths, W)
        elif (calcType == 1):
            G = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(2*pos[:,1])**2/(sig_las**2*c**2))*(sps.erf(curtime*np.sqrt(2)/sig_las) - sps.erf((curtime+t_step)*np.sqrt(2)/sig_las))
            omeg_las_sq = w0**2*(1+pos[:,1]**2/z0**2)
            rho_xy_sq = pos[:,0]**2 + pos[:,2]**2
            W = e**2/2/hbar/mass_e*norm_factor_array**2*w0**2/omeg_las_sq*np.exp(-2*rho_xy_sq/omeg_las_sq)*G
            W = abs(W).real

            prob_vals = np.zeros((spots.shape[0],num_electron_MC_trial))

            prob_vals = sgl_KD_model_caller(prob_vals, spots, paths, W)
        elif (calcType == 2):
            G_x = sig_las*np.sqrt(np.pi/2)*np.exp(-(2*pos[:,0])**2/(sig_las**2*c**2))*(sps.erf(curtime*np.sqrt(2)/sig_las) - sps.erf((curtime+t_step)*np.sqrt(2)/sig_las))
            omeg_las_sq_x = w0**2*(1+pos[:,0]**2/z0**2)
            rho_zy_sq = pos[:,1]**2 + pos[:,2]**2
            W_x = e**2/2/hbar/mass_e*norm_factor_array**2*w0**2/omeg_las_sq_x*np.exp(-2*rho_zy_sq/omeg_las_sq_x)*G_x
            
            G_z = sig_las*np.sqrt(np.pi/2)*np.exp(-(2*pos[:,1])**2/(sig_las**2*c**2))*(sps.erf(curtime*np.sqrt(2)/sig_las) - sps.erf((curtime+t_step)*np.sqrt(2)/sig_las))
            omeg_las_sq_z = w0**2*(1+pos[:,1]**2/z0**2)
            rho_xy_sq = pos[:,0]**2 + pos[:,2]**2
            W_z = e**2/2/hbar/mass_e*norm_factor_array**2*w0**2/omeg_las_sq_z*np.exp(-2*rho_xy_sq/omeg_las_sq_z)*G_z
            
            C0 = -e**2/2/hbar/mass_e*2*norm_factor_array**2*w0**2/np.sqrt(omeg_las_sq_x)/np.sqrt(omeg_las_sq_z)*np.exp(-rho_xy_sq/omeg_las_sq_z - rho_zy_sq/omeg_las_sq_x)
            Tzx = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]-pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]+pos[:,1]-2*c*curtime)/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]+pos[:,1]-2*c*(curtime+t_step))/(sig_las*c*np.sqrt(2))))
            Tznx = -sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]+pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]-pos[:,1]+2*c*curtime)/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]-pos[:,1]+2*c*(curtime+t_step))/(sig_las*c*np.sqrt(2))))
            Tnzx = -sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]+pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((-pos[:,0]+pos[:,1]+2*c*curtime)/(sig_las*c*np.sqrt(2)))-sps.erf((-pos[:,0]+pos[:,1]+2*c*(curtime+t_step))/(sig_las*c*np.sqrt(2))))
            Tnznx = sig_las/2*np.sqrt(np.pi/2)*np.exp(-(pos[:,0]-pos[:,1])**2/(2*sig_las**2*c**2))*(sps.erf((pos[:,0]+pos[:,1]+2*c*curtime)/(sig_las*c*np.sqrt(2)))-sps.erf((pos[:,0]+pos[:,1]+2*c*(curtime+t_step))/(sig_las*c*np.sqrt(2))))

            dGx = np.exp(-2*(pos[:,0]-c*(curtime + t_step))**2/(c**2*sig_las**2)) - np.exp(-2*(pos[:,0]-c*(curtime))**2/(c**2*sig_las**2))
            dWx = -e**2/hbar/mass_e*2*norm_factor_array**2*w0**2/omeg_las_sq_x*np.exp(-2*rho_zy_sq/omeg_las_sq_x)*dGx

            dGz = np.exp(-2*(pos[:,1]-c*(curtime + t_step))**2/(c**2*sig_las**2)) - np.exp(-2*(pos[:,1]-c*(curtime))**2/(c**2*sig_las**2))
            dWz = -e**2/hbar/mass_e*2*norm_factor_array**2*w0**2/omeg_las_sq_x*np.exp(-2*rho_zy_sq/omeg_las_sq_x)*dGx

            dTzx = np.exp(-(pos[:,0] - pos[:,1])**2/(2*c**2*sig_las**2))*(np.exp(-(pos[:,0] + pos[:,1] - 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)) - np.exp(-(pos[:,0] + pos[:,1] - 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)))
            dTnznx = np.exp(-(pos[:,0] - pos[:,1])**2/(2*c**2*sig_las**2))*(np.exp(-(pos[:,0] + pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)) - np.exp(-(pos[:,0] + pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)))
            dTznx = np.exp(-(pos[:,0] + pos[:,1])**2/(2*c**2*sig_las**2))*(np.exp(-(pos[:,0] - pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)) - np.exp(-(pos[:,0] - pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)))
            dTnzx = np.exp(-(pos[:,0] + pos[:,1])**2/(2*c**2*sig_las**2))*(np.exp(-(-pos[:,0] + pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)) - np.exp(-(-pos[:,0] + pos[:,1] + 2*c*(curtime + t_step))**2/(2*c**2*sig_las**2)))

            W_x = W_x.real
            dWx = dWx.real
            W_z = W_z.real
            dWz = dWz.real
            firstDiag = (C0*(Tzx+Tnznx)).real
            dFirstDiag = (C0*(dTzx+dTnznx)).real
            secondDiag = (C0*(Tznx+Tnzx)).real
            dSecondDiag = (C0*(dTznx+dTnzx)).real

            prob_vals = np.zeros((spots.shape[0],num_electron_MC_trial))

            prob_vals = dbl_KD_model_caller(prob_vals, spots, paths, W_x, W_z, firstDiag, secondDiag, dWx, dWz, dFirstDiag, dSecondDiag)
    
        prob_vals = np.absolute(prob_vals)
        prob_val_cumul = np.zeros(prob_vals.shape)
        indices = np.zeros(num_electron_MC_trial, dtype=int)
        
        for j in np.arange(num_electron_MC_trial):
            prob_val_cumul[:,j] = np.cumsum(prob_vals[:,j]/sum(prob_vals[:,j]))
            indices[j] = np.nonzero(prob_val_cumul > np.full(prob_val_cumul.shape,random_sel[j]))[0][0]
        
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

        if (curtime <= 0 and curtime+t_step >= 0):
            print(prob_vals)
        curtime = curtime + t_step
        pbar.update(1)

    pbar.close()

    #print(phase_arr)
    print(np.mean(phase_arr))
    print(np.std(phase_arr))

    return phase_arr, pos, energ_arr, vel_arr

def main(argv):
    time_arr = np.arange(12,26)
    for i in time_arr:
        [phase_arr, pos, energ_arr, vel_arr] = PPPP_calculator(calcType=0,t_mag=i)
                
        data_dump = []
        data['phase_arr'] = phase_arr.tolist()
        data['pos'] = pos.tolist()
        data['energ_arr'] = energ_arr.tolist()
        data['vel_arr'] = vel_arr.tolist()
        data_dump.append(data)
        filename = "crs_" + "t_" + str(i) + ".json"
        with open(filename, "w") as outfile:
            json_data = json.dump(data,outfile)
    '''
    vel = 8.15e7 # m/s
    sig_las = 350e-15 # s
    shift_mag = sig_las*vel # m

    y_shifts = [-1,0,1]*shift_mag; # m
    # cross KD points
    x_shifts = [-100,-50,-50,-50,0,0,0,50]*1e-6 # m
    z_shifts = [0,50,0,-50,0,-50,-100,-50]*1e-6 # m

    [x_mesh, y_mesh] = meshgrid(x_shifts, y_shifts)
    [z_mesh, y_mesh] = meshgrid(z_shifts, y_shifts)
    x_arr = x_mesh.flatten()
    z_arr = z_mesh.flatten()
    y_arr = y_mesh.flatten()

    for i in np.arange(x_arr.size):
        for j in np.arange(z_arr.size):
            for k in np.arange(y_arr.size):
                [phase_arr, pos, energ_arr, vel_arr] = PPPP_calculator(pos_adj_x=x_arr[i],pos_adj_y=y_arr[k],pos_adj_z=z_arr[j],calcType=0)
                
                data_dump = []
                data['phase_arr'] = phase_arr.tolist()
                data['pos'] = pos.tolist()
                data['energ_arr'] = energ_arr.tolist()
                data['vel_arr'] = vel_arr.tolist()
                data_dump.append(data)
                filename = "crs_" + "x_" + str(int(x_arr[i]/1e-6)) + "_z_" + str(int(z_arr[j]/1e-6)) + "_y_" + str(int(y_arr[k]/shift_mag)) + ".json"
                with open(filename, "w") as outfile:
                    json_data = json.dump(data,outfile)

    # single KD points
    x_shifts = [-100,-50,-50,0,0,0]
    z_shifts = [0,50,0,100,50,0]

    [x_mesh, y_mesh] = meshgrid(x_shifts, y_shifts)
    [z_mesh, y_mesh] = meshgrid(z_shifts, y_shifts)
    x_arr = x_mesh.flatten()
    z_arr = z_mesh.flatten()
    y_arr = y_mesh.flatten()

    for i in np.arange(x_arr.size):
        for j in np.arange(z_arr.size):
            for k in np.arange(y_arr.size):
                PPPP_calculator(pos_adj_x=x_arr[i],pos_adj_y=y_arr[k],pos_adj_z=z_arr[j],calcType=1)
                
                data_dump = []
                data['phase_arr'] = phase_arr.tolist()
                data['pos'] = pos.tolist()
                data['energ_arr'] = energ_arr.tolist()
                data['vel_arr'] = vel_arr.tolist()
                data_dump.append(data)
                filename = "sgl_" + "x_" + str(int(x_arr[i]/1e-6)) + "_z_" + str(int(z_arr[j]/1e-6)) + "_y_" + str(int(y_arr[k]/shift_mag)) + ".json"
                with open(filename, "w") as outfile:
                    json_data = json.dump(data,outfile)
                full_elapsed = time.time() - full_init
                print(full_elapsed)

    # double KD points
    x_shifts = [-100,-50,-50,0]
    z_shift = [0,50,0,0]

    [x_mesh, y_mesh] = meshgrid(x_shifts, y_shifts)
    [z_mesh, y_mesh] = meshgrid(z_shifts, y_shifts)
    x_arr = x_mesh.flatten()
    z_arr = z_mesh.flatten()
    y_arr = y_mesh.flatten()

    for i in np.arange(x_arr.size):
        for j in np.arange(z_arr.size):
            for k in np.arange(y_arr.size):
                PPPP_calculator(pos_adj_x=x_arr[i],pos_adj_y=y_arr[k],pos_adj_z=z_arr[j],calcType=2)
                
                data_dump = []
                data['phase_arr'] = phase_arr.tolist()
                data['pos'] = pos.tolist()
                data['energ_arr'] = energ_arr.tolist()
                data['vel_arr'] = vel_arr.tolist()
                data_dump.append(data)
                filename = "dbl_" + "x_" + str(int(x_arr[i]/1e-6)) + "_z_" + str(int(z_arr[j]/1e-6)) + "_y_" + str(int(y_arr[k]/shift_mag)) + ".json"
                with open(filename, "w") as outfile:
                    json_data = json.dump(data,outfile)
                full_elapsed = time.time() - full_init
                print(full_elapsed)
    '''

    #sys.exit(appctxt.app.exec())

if __name__ == '__main__':
    main(sys.argv)

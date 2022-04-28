import numpy as np
from numba import jit, prange
#import cupy as cp
import scipy.integrate as spi
import scipy.interpolate as spip
import math as math
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import tqdm
import sys
from functools import partial
import json

@jit(nopython = True)
def temporal_gauss(z,t,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    val = np.exp(-(2*(z-c*t)**2)/(sig_las**2*c**2))
    return val

@jit(nopython = True)
def omeg_las_sq(z, beam_waist):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    lam = 500 # nm
    w0 = beam_waist # nm
    z0 = np.pi*w0**2/lam # Rayleigh range, nm
    val = w0**2*(1+z**2/z0**2)
    return val

@jit(nopython = True)
def spatial_gauss(rho_xy,z,t, beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    val = 1/np.pi/omeg_las_sq(z, beam_waist)*np.exp(-(2*rho_xy**2)/(omeg_las_sq(z, beam_waist)/temporal_gauss(z,t,sig_las)))
    return val

@jit(nopython = True)
def laser(rho_xy,z,t, beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    val = spatial_gauss(rho_xy,z,t, beam_waist,sig_las)*temporal_gauss(z,t,sig_las)
    return val

def norm_laser_integrand(rho_xy,z,t,beam_waist,sig_las):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    val = 2*np.pi*rho_xy*laser(rho_xy,z,t, beam_waist,sig_las)
    return val

def laser_sum(t, gauss_limit, sig_las, beam_waist):
    planck = 6.626e-34*1e9*1e12 #nJ.*ps

    ## Construct laser beam
    c = 2.9979e8*1e9*1e-12 # nm./ps
    val = spi.dblquad(norm_laser_integrand, -gauss_limit*sig_las + c*t, gauss_limit*sig_las + c*t, 0, gauss_limit*np.sqrt(omeg_las_sq(c*t, beam_waist)), args=[t, beam_waist,sig_las])
    return val

@jit(nopython = True)
def trapz_self(x,y):
    x_diff = x[1:-1] - x[0:-2]
    y_trap = (y[1:-1] + y[0:-2])/2
    return sum(x_diff*y_trap)

@jit(nopython = True)
def feynman_single_calc_func(cur_voxel,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,beam_waist,sig_las,theta,beta,c,lam,t_range,hbar,alpha,mass_e,zshift,xshift):
    #pbar = pass_list[12]
    # Determine slice level --> will determine weighting at the end

    # Assumption: all electrons pass through (x0,z0) at crossover most
    # likely incorrect, but we have nothing else to go off of will only
    # slightly cause the CTF to show a higher than normal resolution

    # reference current voxel xz position grid from travel path
    # calculation and current time

    # calculate photon densities at position grid

    # calculate path for current x(t), y(t), z(t) for specific slice, and
    # voxel m,n. This is the path of the electron, but these values
    # are placed into the laser equation.

    y_vals = init_y_vals[cur_voxel] - vel*t_range
    x_vals = y_vals*x_slopes[cur_voxel] + xshift
    z_vals = y_vals*z_slopes[cur_voxel] + zshift
    rho_vals = np.sqrt(x_vals**2+y_vals**2)
    rho_vals_2 = np.sqrt(z_vals**2+y_vals**2)
    density_vals = norm_factor_array*laser(rho_vals,z_vals,t_range, beam_waist,sig_las)
    full_vals = hbar*alpha*density_vals*lam/np.sqrt(mass_e**2.*(1+vel**2/c**2))
    calc = trapz_self(t_range,full_vals)
    return (not math.isnan(calc))*calc

@jit(nopython = True)
def feynman_double_calc_func(cur_voxel,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,beam_waist,sig_las,theta,beta,c,lam,t_range,hbar,alpha,mass_e,zshift,xshift):
    #pbar = pass_list[12]
    # Determine slice level --> will determine weighting at the end

    # Assumption: all electrons pass through (x0,z0) at crossover most
    # likely incorrect, but we have nothing else to go off of will only
    # slightly cause the CTF to show a higher than normal resolution

    # reference current voxel xz position grid from travel path
    # calculation and current time

    # calculate photon densities at position grid

    # calculate path for current x(t), y(t), z(t) for specific slice, and
    # voxel m,n. This is the path of the electron, but these values
    # are placed into the laser equation.

    y_vals = init_y_vals[cur_voxel] - vel*t_range
    x_vals = y_vals*x_slopes[cur_voxel] + xshift
    z_vals = y_vals*z_slopes[cur_voxel] + zshift
    rho_vals = np.sqrt(x_vals**2+y_vals**2)
    rho_vals_2 = np.sqrt(z_vals**2+y_vals**2)
    density_vals = norm_factor_array*laser(rho_vals,z_vals,t_range, beam_waist,sig_las) + norm_factor_array*laser(rho_vals_2,x_vals,t_range, beam_waist,sig_las)
    full_vals = hbar*alpha*density_vals*lam/np.sqrt(mass_e**2.*(1+vel**2/c**2))
    calc = trapz_self(t_range,full_vals)
    return (not math.isnan(calc))*calc

@jit(nopython = True)
def quasi_single_calc_func(cur_voxel,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,beam_waist,sig_las,theta,beta,c,lam,t_range,hbar,alpha,mass_e,zshift,xshift):
    #pbar = pass_list[12]
    # Determine slice level --> will determine weighting at the end

    # Assumption: all electrons pass through (x0,z0) at crossover most
    # likely incorrect, but we have nothing else to go off of will only
    # slightly cause the CTF to show a higher than normal resolution

    # reference current voxel xz position grid from travel path
    # calculation and current time

    # calculate photon densities at position grid

    # calculate path for current x(t), y(t), z(t) for specific slice, and
    # voxel m,n. This is the path of the electron, but these values
    # are placed into the laser equation.

    y_vals = init_y_vals[cur_voxel] - vel*t_range
    x_vals = y_vals*x_slopes[cur_voxel] + xshift
    z_vals = y_vals*z_slopes[cur_voxel] + zshift
    rho_vals = np.sqrt(x_vals**2+y_vals**2)
    full_vals = (norm_factor_array*laser(rho_vals,z_vals,t_range, beam_waist,sig_las))**2*(1-beta**2*np.cos(2*np.pi*(z_vals-c*t_range)/lam)**2*np.cos(theta)**2)
    calc = trapz_self(t_range,full_vals)
    return (not math.isnan(calc))*calc

@jit(nopython = True)
def quasi_double_calc_func(cur_voxel,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,beam_waist,sig_las,theta,beta,c,lam,t_range,hbar,alpha,mass_e,zshift,xshift):
    #pbar = pass_list[12]
    # Determine slice level --> will determine weighting at the end

    # Assumption: all electrons pass through (x0,z0) at crossover most
    # likely incorrect, but we have nothing else to go off of will only
    # slightly cause the CTF to show a higher than normal resolution

    # reference current voxel xz position grid from travel path
    # calculation and current time

    # calculate photon densities at position grid

    # calculate path for current x(t), y(t), z(t) for specific slice, and
    # voxel m,n. This is the path of the electron, but these values
    # are placed into the laser equation.

    y_vals = init_y_vals[cur_voxel] - vel*t_range
    x_vals = y_vals*x_slopes[cur_voxel] + xshift
    z_vals = y_vals*z_slopes[cur_voxel] + zshift
    rho_vals = np.sqrt(x_vals**2+y_vals**2)
    rho_vals_2 = np.sqrt(z_vals**2+y_vals**2)
    full_vals = (norm_factor_array*laser(rho_vals,z_vals,t_range, beam_waist,sig_las))**2*(1-beta**2*np.cos(2*np.pi*(z_vals-c*t_range)/lam)**2*np.cos(theta)**2) + (norm_factor_array*laser(rho_vals_2,z_vals,t_range, beam_waist,sig_las))**2*(1-beta**2*np.cos(2*np.pi*(x_vals-c*t_range)/lam)**2*np.cos(theta)**2)
    calc = trapz_self(t_range,full_vals)
    return (not math.isnan(calc))*calc

# electron beam functions
def omeg_ebeam(y,xover_slope):
    val = np.absolute(xover_slope*y)
    return val

def e_beam_xz(rho_xz,size_direct_beam):
    val = 1/np.pi/(size_direct_beam)**2*np.exp(-(2*rho_xz**2)/(size_direct_beam)**2)
    return val

def e_beam_xz_raster(x,z,size_direct_beam):
    val = 1/np.pi/(size_direct_beam)**2*np.exp(-(2*(x**2 + z**2))/(size_direct_beam)**2)
    return val

def e_beam_yt(y,sig_ebeam,vel):
    val = 1/np.pi/sig_ebeam**2/vel**2*np.exp(-(2*(y)**2)/(sig_ebeam**2*vel**2))
    return val

@jit(nopython = True,parallel = True)
def model_caller(init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift,calc_type,laser_num,num_voxels):
    voxel_grid_phase_data_res = np.zeros(num_voxels)
    for i in prange(num_voxels):
        try:
            if calc_type == 0:
                if laser_num == 1:
                    voxel_grid_phase_data_res[i] = quasi_single_calc_func(i,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift)
                elif laser_num == 2:
                    voxel_grid_phase_data_res[i] = quasi_double_calc_func(i,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift)
            elif calc_type == 1:
                if laser_num == 1:
                    voxel_grid_phase_data_res[i] = feynman_single_calc_func(i,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift)
                elif laser_num == 2:
                    voxel_grid_phase_data_res[i] = feynman_double_calc_func(i,init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift)
        except:
            print('error in calculating slice integral')

    return voxel_grid_phase_data_res

def PPPP_calculator(calc_type=0,laser_num=1,ebeam_type=0,sig_ebeam=1,sig_las=1,w0=100e3,E_pulse=1,voxel_granularity=9,slice_granularity=9,focus_granularity=1,num_points_to_add=2000,size_direct_beam=100e3,gauss_limit=3):
    print('Seeding workspace with relevant information.')

    # Initializing fundamental constants
    planck = 6.626e-34*1e9*1e12 #nJ.*ps
    hbar = planck/2/np.pi
    alpha = 1/137 # fine structure constant
    mass_e = 9.11e-31*1e15 # pg

    c = 2.9979e8*1e9*1e-12 # nm./ps
    lam = 500 # nm
    theta = 0

    xover_slope = 3e-3/300e-3 # 3 mm in 300 mm of travel
    xover_angle = np.arctan(xover_slope) # degrees
    vel = 2.33e8*1e9/1e12 # velocity of electron, 200 kV, nm./ps
    beta = vel/c

    # Initializing simulation parameters
    # voxel_granularity
    # slice_granularity
    # focus_granularity
    # gauss_limit

    # size_direct_beam, ebeam radius in nm

    # calc_type, 1 for quasiclassical, 0 for feynman
    # laser_num, 1 for single laser, 2 for double laser
    # ebeam_type, 0 for pulsed, 1 for uniform

    #sig_ebeam, time resolution of ebeam, ps
    #sig_las, ps
    #w0, nm

    E_photon = planck*c/lam # nJ
    # E_pulse # nJ
    n_pulse = E_pulse/E_photon # number of photons per pulse

    data = {}
    data['model'] = calc_type
    data['laser_geometry'] = laser_num
    data['ebeam_geometry'] = ebeam_type
    if (ebeam_type == 0):
        data['sig_ebeam'] = sig_ebeam
        data['sig_las'] = sig_las
    else:
        data['sig_ebeam'] = sig_las
        data['sig_las'] = sig_las
    data['beam_waist'] = w0
    data['E_pulse'] = E_pulse

    print('preparing electron slices')

    # Beginning slice separation in time and space (y-direction)
    sig_ratios = np.array([-gauss_limit, -1.6449, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, -0.0627]) # limit, 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
    weights = np.array([0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025])

    if (sig_ebeam < sig_las):
        t_bounds_overall = []
        sig_ratios = np.insert(sig_ratios,0,-gauss_limit*sig_las/sig_ebeam)
        weights = np.insert(weights,0,0.005)
        num_voxels = 0
        for i in np.arange(len(t_bounds_overall)-1):
            t_bounds_add = np.linspace(sig_ratios[i]*sig_ebeam,sig_ratios[i+1]*sig_ebeam,round(weights[i]*slice_granularity))
            t_bounds_overall.append(t_bounds_add[1:-2])
            num_voxels = num_voxels + len(t_bounds_overall[i])

        voxels_left = slice_granularity + 1 - num_voxels*2
        t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_ebeam,-sig_ratios[-1]*sig_ebeam,voxels_left))
    elif (sig_las < sig_ebeam):
        t_bounds_overall = []
        sig_ratios = np.insert(sig_ratios,0,-gauss_limit*sig_las/sig_ebeam)
        weights = np.insert(weights,0,0.005)
        num_voxels = 0
        for i in np.arange(len(t_bounds_overall)-1):
            t_bounds_add = np.linspace(sig_ratios[i]*sig_las,sig_ratios[i+1]*sig_las,round(weights[i]*slice_granularity))
            t_bounds_overall.append(t_bounds_add[1:-2])
            num_voxels = num_voxels + len(t_bounds_overall[i])

        voxels_left = slice_granularity + 1 - num_voxels*2
        t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_las,-sig_ratios[-1]*sig_las,voxels_left))
    else:
        weights[1] = 0.075
        t_bounds_overall = []
        num_voxels = 0
        for i in np.arange(len(t_bounds_overall)-1):
            t_bounds_add = np.linspace(sig_ratios[i]*sig_ebeam,sig_ratios[i+1]*sig_ebeam,round(weights[i]*slice_granularity))
            t_bounds_overall.append(t_bounds_add[1:-2])
            num_voxels = num_voxels + len(t_bounds_overall[i])

        voxels_left = slice_granularity + 1 - num_voxels*2
        t_bounds_overall.append(np.linspace(sig_ratios[-1]*sig_ebeam,-sig_ratios[-1]*sig_ebeam,voxels_left))

    t_bounds = np.array([])
    for i in np.arange(len(t_bounds_overall)):
        t_bounds = np.append(t_bounds,t_bounds_overall[i])

    for i in np.flip(np.arange(len(t_bounds_overall)-1)):
        t_bounds = np.append(t_bounds,np.flip(-t_bounds_overall[i]))

    y_bounds = t_bounds*vel

    print('calculating the electron beam normalization constants')
    # normalizing the electron beam in the xz plane (orthogonal to principal electron-optical axis) and the y-direction
    e_beam_xz_integral = spi.quad(lambda x: 2*np.pi*x*e_beam_xz(x,size_direct_beam), 0, gauss_limit*size_direct_beam)
    e_beam_xz_norm = 1/e_beam_xz_integral[0]

    if (ebeam_type == 0):
        e_beam_int = spi.quad(lambda y: e_beam_yt(y,sig_ebeam,vel), y_bounds[1], y_bounds[-1])
        e_beam_yt_norm = 1 / e_beam_int[0]
    elif (ebeam_type == 1):
        e_beam_int = 1/(y_bounds[-1] - y_bounds[1])

    # setting up data matrices
    voxel_grid_cumulative_phase_data = np.zeros((focus_granularity, focus_granularity))
    curwl_matrix = np.zeros((focus_granularity, focus_granularity))
    curwh_matrix = np.zeros((focus_granularity, focus_granularity))
    curql_matrix = np.zeros((focus_granularity, focus_granularity))
    curqh_matrix = np.zeros((focus_granularity, focus_granularity))
    voxel_grid_phase_data = np.zeros((voxel_granularity,voxel_granularity,slice_granularity))
    voxel_grid_slope_x_data = np.zeros((voxel_granularity,voxel_granularity)) # travel path information for each voxel, only need one representative plane for each
    voxel_grid_slope_z_data = np.zeros((voxel_granularity,voxel_granularity)) # travel path information for each voxel, only need one representative plane for each
    voxel_grid_y_data = np.zeros((voxel_granularity,voxel_granularity,slice_granularity))

    # setting up y-direction and t-direction range
    y_range = np.zeros(slice_granularity)
    t_range = np.zeros(slice_granularity)

    for l in np.arange(slice_granularity):
        voxel_grid_y_data[:,:,l] = np.ones((voxel_granularity,voxel_granularity))*((y_bounds[l]+y_bounds[l+1])/2)
        y_range[l] = (y_bounds[l]+y_bounds[l+1])/2
        t_range[l] = (t_bounds[l]+t_bounds[l+1])/2

    print('expanding the integral space')
    # setting up temporal calculation array for trapz. increases the granularity with a focus on the central region
    min_t_vals_sep = num_points_to_add
    num_points_add = lambda t: - 1 / (1 + np.exp(-np.absolute(t) / (5 * t_range[-1]))) + 1
    add_points = np.round(min_t_vals_sep * num_points_add(t_bounds)).astype(int)
    tot_points = np.sum(add_points).astype(int)
    t_range_extended = np.zeros(tot_points - add_points[-1])
    tot_added = 0
    for l in np.arange(t_bounds.size-1):
        t_range_extended[tot_added:tot_added + add_points[l]] = np.linspace(t_bounds[l], t_bounds[l + 1],add_points[l])
        tot_added = tot_added + add_points[l]

    print('calculating slopes of travel')
    # setting up slopes for the electrons to take in the x and z directions as a function of initial position
    if (sig_ebeam < sig_las):
        y_dist_from_center = gauss_limit*sig_las*vel
        for j in np.arange(voxel_granularity):
            voxel_grid_slope_x_data[j,:] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel,xover_slope),gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel,xover_slope),voxel_granularity)/y_dist_from_center
            voxel_grid_slope_z_data[:,j] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel,xover_slope),gauss_limit*omeg_ebeam(gauss_limit*sig_las*vel,xover_slope),voxel_granularity)/y_dist_from_center
    elif (sig_las <= sig_ebeam):
        y_dist_from_center = gauss_limit*sig_ebeam*vel
        for j in np.arange(voxel_granularity):
            voxel_grid_slope_x_data[j,:] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel,xover_slope),gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel,xover_slope),voxel_granularity)/y_dist_from_center
            voxel_grid_slope_z_data[:,j] = np.linspace(-gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel,xover_slope),gauss_limit*omeg_ebeam(gauss_limit*sig_ebeam*vel,xover_slope),voxel_granularity)/y_dist_from_center

    print('weighting electron beam in the y direction')
    ## calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
    # voxel_xz_grid_weights = zeros(voxel_granularity, voxel_granularity);
    voxel_y_weights = np.zeros(slice_granularity)

    for l in np.arange(slice_granularity):
        if (ebeam_type == 0):
            weight_int = spi.quad(lambda y: e_beam_yt(y,sig_ebeam,vel), y_bounds[l], y_bounds[l + 1])
            voxel_y_weights[l] = e_beam_yt_norm*weight_int[0]
        elif (ebeam_type == 1):
            voxel_y_weights[l] = e_beam_yt_norm*(y_bounds[l + 1] - y_bounds[l])

    print('normalizing the laser beam over time')
    ## establishing interpolated normalization for laser
    laser_sum_array = np.zeros_like(t_bounds)
    laser_sum_err = np.zeros_like(t_bounds)

    #input('beginning integral')
    for i in np.arange(t_bounds.size):
        val_int = laser_sum(t_bounds[i], gauss_limit, sig_las, w0)
        laser_sum_array[i] = val_int[0]
        laser_sum_err[i] = val_int[1]

    select_zero_laser_sum_array = np.where(laser_sum_array == 0, 1, 0)
    norm_factor_small_array = n_pulse/(select_zero_laser_sum_array*1e308 + laser_sum_array)
    norm_laser_interp = spip.interp1d(t_bounds, norm_factor_small_array, kind='cubic')
    norm_factor_array = norm_laser_interp(t_range_extended)

    print('normalizing the electron beam in the xz plane')
    # calculating electron beam xz weights
    voxel_xz_focus_weights = np.zeros((focus_granularity, focus_granularity))
    num_condense = np.zeros((focus_granularity, focus_granularity))
    xz_spacing = 2 * (size_direct_beam) / voxel_granularity
    xz_centers = np.linspace(-size_direct_beam - xz_spacing / 2, size_direct_beam + xz_spacing / 2, voxel_granularity)
    xz_bounds = np.linspace(-size_direct_beam - xz_spacing / 2, size_direct_beam + xz_spacing / 2,voxel_granularity + 1)
    if (focus_granularity != 1):
        xz_focus_centers = np.linspace(-size_direct_beam - xz_spacing / 2, size_direct_beam + xz_spacing / 2,focus_granularity)
        xz_focus_bounds = np.linspace(-size_direct_beam - xz_spacing / 2, size_direct_beam + xz_spacing / 2,focus_granularity + 1)
        voxel_xz_grid_weights = np.zeros((voxel_granularity, voxel_granularity))
        for j in np.arange(voxel_granularity):
            for k in np.arange(voxel_granularity):
                cur_xz_integral = spi.dblquad(lambda x,z: e_beam_xz_raster(x,z,size_direct_beam), xz_bounds[j], xz_bounds[j + 1], xz_bounds[k],xz_bounds[k + 1])
                voxel_xz_grid_weights[j, k] = e_beam_xz_norm * cur_xz_integral[0]
    else:
        xz_focus_centers = 0
        xz_focus_bounds = 0
        voxel_xz_grid_weights = np.ones((voxel_granularity, voxel_granularity))

    print('calculating the summation for condensing to the focal granularity')
    if (focus_granularity != 1):
        for u in np.arange(focus_granularity):
            for v in np.arange(focus_granularity):
                curu1 = xz_focus_bounds[u]
                curu2 = xz_focus_bounds[u + 1]
                curv1 = xz_focus_bounds[v]
                curv2 = xz_focus_bounds[v + 1]

                curwh = voxel_granularity
                foundwh = 0
                curwl = 1
                foundwl = 0
                curqh = voxel_granularity
                foundqh = 0
                curql = 1
                foundql = 0
                for q in np.arange(xz_centers.size):
                    if ((xz_centers[q] <= curu2) and (foundwh != 1 and foundwl == 1)):
                        curwh = q
                        foundwh = 1
                    elif (xz_centers[q] >= curu1 and foundwl != 1):
                        foundwl = 1
                        curwl = q

                    if (xz_centers[q] <= curv2 and foundqh != 1 and foundql == 1):
                        curqh = q
                        foundqh = 1
                    elif (xz_centers[q] >= curv1 and foundql != 1):
                        foundql = 1
                        curql = q

                curwl_matrix[u, v] = curwl
                curwh_matrix[u, v] = curwh
                curql_matrix[u, v] = curql
                curqh_matrix[u, v] = curqh

        for u in np.arange(focus_granularity):
            for v in np.arange(focus_granularity):
                curwl = curwl_matrix[u, v].astype(int)
                curwh = curwh_matrix[u, v].astype(int)
                curql = curql_matrix[u, v].astype(int)
                curqh = curqh_matrix[u, v].astype(int)
                num_condense[u, v] = np.absolute((curwh - curwl + 1) * (curqh - curql + 1))
                voxel_xz_focus_weights[u, v] = sum(sum(voxel_xz_grid_weights[curwl:curwh, curql:curqh]))
    else:
        num_condense[0,0] = 1
        voxel_xz_focus_weights[0,0] = 1
        curwl_matrix[0,0] = 0
        curwh_matrix[0,0] = voxel_granularity-1
        curql_matrix[0,0] = 0
        curqh_matrix[0,0] = voxel_granularity-1


    ## Loop

    if (focus_granularity != 1):
        z_drift = np.linspace(-size_direct_beam, size_direct_beam, focus_granularity)
        x_drift = np.linspace(-size_direct_beam, size_direct_beam, focus_granularity)
    else:
        z_drift = np.array([0])
        x_drift = np.array([0])

    num_voxels = slice_granularity*voxel_granularity**2

    init_y_vals = np.zeros(num_voxels)
    x_slopes = np.zeros(num_voxels)
    z_slopes = np.zeros(num_voxels)
    for cur_voxel in np.arange(num_voxels):
        cur_slice = math.floor(cur_voxel/voxel_granularity**2)
        cur_voxel_num = cur_voxel % voxel_granularity**2
        m = math.floor(cur_voxel_num/voxel_granularity)
        n = cur_voxel_num % voxel_granularity

        init_y_vals[cur_voxel] = voxel_grid_y_data[m,n,cur_slice]
        x_slopes[cur_voxel] = voxel_grid_slope_x_data[m,n]
        z_slopes[cur_voxel] = voxel_grid_slope_z_data[m,n]

    pbar = tqdm.tqdm(total=z_drift.size)
    for zin in np.arange(z_drift.size):

        for xin in np.arange(x_drift.size):

            t = time.time()

            z_shift = z_drift[zin]
            x_shift = x_drift[xin]

            '''
            method_list = list()
            method_list.append(init_y_vals)
            method_list.append(x_slopes)
            method_list.append(z_slopes)
            method_list.append(norm_factor_array)
            method_list.append(vel)
            method_list.append(w0)
            method_list.append(sig_las)
            method_list.append(theta)
            method_list.append(beta)
            method_list.append(c)
            method_list.append(lam)
            method_list.append(t_range_extended)
            method_list.append(hbar)
            method_list.append(alpha)
            method_list.append(mass_e)
            method_list.append(z_shift)
            method_list.append(x_shift)
            '''

            voxel_grid_phase_data_unpacked = model_caller(init_y_vals,x_slopes,z_slopes,norm_factor_array,vel,w0,sig_las,theta,beta,c,lam,t_range_extended,hbar,alpha,mass_e,z_shift,x_shift,calc_type,laser_num,num_voxels)

            #print('Sending results from GPU to CPU')
            #voxel_grid_phase_data_unpacked = cp.array(voxel_grid_phase_data_res).get()

            #print('Successful transfer')
            # generate map distribution of electron beam, summing and averaging over
            # all slices

            for cur_voxel in np.arange(num_voxels):
                cur_slice = math.floor(cur_voxel/voxel_granularity**2)
                cur_voxel_num = cur_voxel % voxel_granularity**2
                m = math.floor(cur_voxel_num/voxel_granularity)
                n = cur_voxel_num % voxel_granularity
                voxel_grid_phase_data[m,n,cur_slice] = voxel_grid_phase_data_unpacked[cur_voxel]

            final_phase_data = np.zeros((voxel_granularity, voxel_granularity))

            for m in np.arange(voxel_granularity):
              for n in np.arange(voxel_granularity):
                  for p in np.arange(slice_granularity):
                      final_phase_data[m,n] = final_phase_data[m,n] + voxel_grid_phase_data[m,n,p]*voxel_y_weights[p]

            if (focus_granularity != 1):
                for u in np.arange(focus_granularity):
                    for v in np.arange(focus_granularity):
                        curwl = curwl_matrix[u, v].astype(int)
                        curwh = curwh_matrix[u, v].astype(int)
                        curql = curql_matrix[u, v].astype(int)
                        curqh = curqh_matrix[u,v].astype(int)
                        voxel_grid_cumulative_phase_data[u, v] = voxel_grid_cumulative_phase_data[u, v] + sum(sum(final_phase_data[curwl:curwh, curql:curqh])) / num_condense[u, v] * voxel_xz_focus_weights[zin, xin]
            else:
                voxel_grid_cumulative_phase_data = final_phase_data

        pbar.update(1)
    pbar.close()
    elapsed = time.time() - t
    print(elapsed)

    data['results'] = voxel_grid_cumulative_phase_data.tolist()
    return data

def main(argv):
    #mempool = cp.get_default_memory_pool()

    #with cp.cuda.Device(0):
    #    mempool.set_limit(fraction = 0.75)  # 75% gpu usage in memory

    voxel_granularity = 49
    slice_granularity = 49
    focus_granularity = 1
    num_points_to_add = 2000
    gauss_limit = 3

    size_direct_beam = 100e3 # ebeam radius in nm

    calc_type = 1 # 1 for quasiclassical, 0 for feynman
    laser_num = 2 # 1 for single laser, 2 for double laser
    ebeam_type = 0 # 0 for pulsed, 1 for uniform

    sig_ebeam = 10 # time resolution of ebeam, ps
    sig_las = 10 # ps
    w0 = 100e3 # nm

    E_pulse = 22.25e3 # nJ
    data = PPPP_calculator(calc_type,laser_num,ebeam_type,sig_ebeam,sig_las,w0,E_pulse,voxel_granularity,slice_granularity,focus_granularity,num_points_to_add,size_direct_beam,gauss_limit)
    data = PPPP_calculator(calc_type, laser_num, ebeam_type, sig_ebeam, sig_las, w0, E_pulse, voxel_granularity, slice_granularity, focus_granularity, num_points_to_add, size_direct_beam, gauss_limit)

    json_data = json.dumps(data)
    with open("sample.json", "w") as outfile:
        outfile.write(json_data)

    fig_data = data['results']
    plt.figure()
    plt.imshow(fig_data)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
    input('awaiting')

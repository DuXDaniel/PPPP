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

def main(argv):
    laser_res=350e-15
    E_pulse=10e-6
    beam_waist=20e-6
    gauss_limit=4
    las_wav=517e-9

    theta =  0

    lam = las_wav # m
    w0 = beam_waist # m
    sig_las = laser_res # s
    z0 = math.pi*w0**2/lam # Rayleigh range, nm

    for i in np.arange(18,25):
        full_init = time.time()
        data = {}

        t_step = 10.**(-1*i)#1e-15 # fs steps
        t_range = np.arange(-gauss_limit*sig_las,gauss_limit*sig_las,t_step)
        norm_factor_array = Norm_Laser_Calculator(t_range,gauss_limit,sig_las,lam,w0,E_pulse)
        
        data_dump = []
        data['norm_factor'] = norm_factor_array.imag.tolist()
        data_dump.append(data)
        filename = "norm_factor_" + str(i) + ".json"
        with open(filename, "w") as outfile:
            json_data = json.dump(data,outfile)
        full_elapsed = time.time() - full_init
        print(full_elapsed)

if __name__ == '__main__':
    main(sys.argv)
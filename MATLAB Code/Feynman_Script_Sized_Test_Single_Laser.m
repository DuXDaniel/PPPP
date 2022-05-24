laser_res_range = 1000;
laser_width_range = 100e3;
e_res_range = 1000;
laser_pulse_range = 1;

[phase] = Simul_Feynman_Func_Sized_V01(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Simul_Feynman_Func_Sized_V01_nslas_nse_100um_1nJ'];
save(save_file_name,'phase');
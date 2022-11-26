laser_res_range = 1e3; % ps
laser_width_range = 100e3; % nm
laser_pulse_range = 1; % nJ

[phase] = Simul_Feynman_Func_Sized_UniformE_V01_SM(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Simul_Feynman_Func_Sized_UniformE_sinlas_nslas_nse_100um_1nJ_SM'];
save(save_file_name,'phase');

[phase] = Simul_Feynman_Func_Sized_Double_Laser_UniformE_V01_SM(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Simul_Feynman_Func_Sized_UniformE_doulas_nslas_nse_100um_1nJ_SM'];
save(save_file_name,'phase');

% have to multiply by % of time laser is in contact with continuous ebeam
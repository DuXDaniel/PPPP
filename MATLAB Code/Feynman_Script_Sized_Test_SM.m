laser_res_range = 1000;
laser_width_range = 100e3;
e_res_range = 1000;
laser_pulse_range = 1;

[phase] = Simul_Feynman_Func_Sized_V01_SM(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Simul_Feynman_Func_Sized_sinlas_nslas_nse_100um_100nJ_SM'];
save(save_file_name,'phase');

[phase] = Simul_Feynman_Func_Sized_Double_Laser_V01_SM(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Simul_Feynman_Func_Sized_doulas_nslas_nse_100um_100nJ_SM'];
save(save_file_name,'phase');
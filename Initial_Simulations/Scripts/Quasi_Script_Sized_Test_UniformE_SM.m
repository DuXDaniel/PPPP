laser_res_range = 10; % ps
laser_width_range = 100e3; % nm
laser_pulse_range = 1;

[phase] = Quasiclassical_Func_Sized_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_sinlas_10pslas_10pse_100um_1_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_doulas_10pslas_10pse_100um_1_SM_trapz'];
save(save_file_name,'phase');

laser_width_range = 1e3; % nm

[phase] = Quasiclassical_Func_Sized_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_sinlas_10pslas_10pse_1um_1_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_doulas_10pslas_10pse_1um_1_SM_trapz'];
save(save_file_name,'phase');

laser_pulse_range = 1000;

[phase] = Quasiclassical_Func_Sized_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_sinlas_10pslas_10pse_1um_1000_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_doulas_10pslas_10pse_1um_1000_SM_trapz'];
save(save_file_name,'phase');

laser_width_range = 100e3; % nm
laser_pulse_range = 1000;

[phase] = Quasiclassical_Func_Sized_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_sinlas_10pslas_10pse_100um_1000_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_UniformE_V01_SM_trapz(laser_width_range, laser_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_UniformE_doulas_10pslas_10pse_100um_1000_SM_trapz'];
save(save_file_name,'phase');

% have to multiply by % of time laser is in contact with continuous ebeam
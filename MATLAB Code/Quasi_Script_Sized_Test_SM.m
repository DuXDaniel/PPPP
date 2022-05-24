laser_res_range = 10;
laser_width_range = 100e3;
e_res_range = 10;
laser_pulse_range = 1;

[phase] = Quasiclassical_Func_Sized_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_sinlas_10pslas_10pse_100um_1_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_doulas_10pslas_10pse_100um_1_SM_trapz'];
save(save_file_name,'phase');

laser_width_range = 1e3;
laser_pulse_range = 1;

[phase] = Quasiclassical_Func_Sized_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_sinlas_10pslas_10pse_1um_1_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_doulas_10pslas_10pse_1um_1_SM_trapz'];
save(save_file_name,'phase');

laser_width_range = 1e3;
laser_pulse_range = 1000;

[phase] = Quasiclassical_Func_Sized_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_sinlas_10pslas_10pse_1um_1000_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_doulas_10pslas_10pse_1um_1000_SM_trapz'];
save(save_file_name,'phase');

laser_width_range = 100e3;
laser_pulse_range = 1000;

[phase] = Quasiclassical_Func_Sized_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_sinlas_10pslas_10pse_100um_1000_SM_trapz'];
save(save_file_name,'phase');

[phase] = Quasiclassical_Func_Sized_Double_Laser_V01_SM_trapz(laser_width_range, laser_res_range, e_res_range, laser_pulse_range);
                
save_file_name = ['Quasiclassical_Func_Sized_doulas_10pslas_10pse_100um_1000_SM_trapz'];
save(save_file_name,'phase');
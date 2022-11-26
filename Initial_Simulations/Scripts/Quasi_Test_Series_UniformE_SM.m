laser_res_range = 10;
laser_width_range = 10;
e_res_range = logspace(1,3,2);
laser_pulse_range = logspace(-3,3,2);

for i = 1:length(laser_res_range)
    for l = 1:length(laser_pulse_range)
        for m = 1:length(laser_width_range)
            [phase] = Quasiclassical_Func_UniformE_V01_SM_trapz(laser_width_range(m), laser_res_range(i), laser_pulse_range(l));
            
            assign_var_name = ['Quasiclassical_Func_UniformE_V01_SM_trapz_',num2str(i),'_',num2str(l),'_',num2str(m)];
            save(assign_var_name,'phase');
            
            [phase] = Quasiclassical_Func_Double_Laser_UniformE_V01_SM_trapz(laser_width_range(m), laser_res_range(i), laser_pulse_range(l));
            
            assign_var_name = ['Quasiclassical_Func_Double_Laser_UniformE_V01_SM_trapz_',num2str(i),'_',num2str(l),'_',num2str(m)];
            save(assign_var_name,'phase');
            disp('m');
            m
        end
        disp('l');
        l
    end
    disp('i');
    i
end
laser_res_range = logspace(3,5,3);
laser_width_range = logspace(3,6,10);
laser_pulse_range = logspace(1,3,10);

for i = 1:length(laser_res_range)
    for l = 1:length(laser_pulse_range)
        for m = 1:length(laser_width_range)
            [phase] = Simul_Feynman_Func_UniformE_V01(laser_width_range(m), laser_res_range(i), laser_pulse_range(l));
            
            assign_var_name = ['Simul_Feynman_Func_V01_',num2str(i),'_',num2str(l),'_',num2str(m)];
            assignin('base',assign_var_name,phase);
            disp('m');
            m
        end
        disp('l');
        l
    end
    disp('i');
    i
end
save_file_name = 'Simul_Feynman_Func_UniformE_V01_Single_Laser_Results';
save(save_file_name);
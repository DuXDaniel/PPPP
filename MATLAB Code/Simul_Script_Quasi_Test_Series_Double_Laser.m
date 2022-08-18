laser_res_range = logspace(1,3,2);
laser_width_range = logspace(3,6,2);
e_res_range = logspace(1,3,2);
laser_pulse_range = logspace(-3,3,3);

for i = 1:length(laser_res_range)
    for j = 1:length(e_res_range)
        for l = 1:length(laser_pulse_range)
            for m = 1:length(laser_width_range)
                [phase] = Quasiclassical_Func_Double_Laser_V01(laser_width_range(m), laser_res_range(i), e_res_range(j), laser_pulse_range(l));
                
                assign_var_name = ['Quasiclassical_Func_Double_Laser_V01_',num2str(i),'_',num2str(j),'_',num2str(l),'_',num2str(m)];
                assignin(assign_var_name,'phase');
                disp('m');
                m
            end
            disp('l');
            l
        end
        disp('j');
        j
    end
    disp('i');
    i
end
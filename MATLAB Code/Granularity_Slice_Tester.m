laser_res_range = logspace(1,3,2);
laser_width_range = logspace(3,6,2);
e_res_range = logspace(1,3,2);
laser_pulse_range = logspace(-3,3,3);

las_res_int = laser_res_range(1);
e_res_int = e_res_range(1);
las_width_int = laser_width_range(1);
las_pulse_int = laser_pulse_range(end);

points_to_add = 2000;

voxel_gran_length = 3:2:11;
slice_gran_length = 3:2:11;
%slice_gran_length = slice_gran_length(1:81);

for k = 1:length(points_to_add)
    for i = 3%1:length(voxel_gran_length)
        for j = 3%1:length(slice_gran_length)
            %         [phase, slices] = Quasiclassical_Func_V01_SM_VarVox(las_width_int, las_res_int, e_res_int, las_pulse_int, voxel_gran_length(i)^2, slice_gran_length(j)^2);
            %
            %         assign_var_name = ['Quasiclassical_Func_V01_SM_VarVox_1_1_1_3_',num2str(i),'_',num2str(j)];
            %         save(assign_var_name,'phase','slices');
            
            [phase, slices] = Quasiclassical_Func_Double_Laser_V01_SM_VarVox(las_width_int, las_res_int, e_res_int, las_pulse_int, voxel_gran_length(i)^2, slice_gran_length(j)^2, points_to_add(k));
%             
%             assign_var_name = ['Quasiclassical_Func_Double_Laser_V01_SM_VarVox_1_1_1_3_',num2str(i), '_',num2str(j), '_', num2str(k)];
%             save(assign_var_name,'phase','slices');
            disp('j');
            j
        end
        disp('i');
        i
    end
end
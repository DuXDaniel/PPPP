function [voxel_grid_phase_data_unpacked] = quasiclassical_trapz_calculation(num_voxels,init_y_vals,x_slopes,z_slopes,t_range,norm_factor_array,laser,norm_func,beta,c,lambda,theta,integral_bound,t_bounds,vel)

% h2 = waitbar(0, 'Calculating phase shift for each slice');

disp('init gpu store');

voxel_grid_phase_data_unpacked = gpuArray(zeros(num_voxels,1));
norm_func_vals = gpuArray(norm_func(t_range));
init_y_vals_gpu = gpuArray(init_y_vals);
x_slopes_gpu = gpuArray(x_slopes);
z_slopes_gpu = gpuArray(z_slopes);
t_range_gpu = gpuArray(t_range);

disp('fin gpu store');

parfor cur_voxel = 1:num_voxels
    y_vals = init_y_vals_gpu(cur_voxel) - vel.*t_range_gpu;
    x_vals = y_vals.*x_slopes_gpu(cur_voxel);
    z_vals = y_vals.*z_slopes_gpu(cur_voxel);
    rho_vals = sqrt(x_vals.^2+y_vals.^2);
    full_vals = (norm_func_vals.*laser(rho_vals,z_vals,t_range_gpu)).^2.*(1-beta.^2.*cos(2.*pi.*(z_vals-c.*t_range_gpu)/lambda).^2.*cos(theta).^2);
    voxel_grid_phase_data_unpacked(cur_voxel) = trapz(t_range_gpu,full_vals);
end

voxel_grid_phase_data_unpacked = gather(voxel_grid_phase_data_unpacked);

% for cur_voxel = 1:num_voxels
%     % calculate path for current x(t), y(t), z(t) for specific slice, and
%     % voxel m,n. This is the path of the electron, but these values
%     % are placed into the laser equation.
%     
%     y_func = @(t) init_y_vals(cur_voxel)-vel.*t;
%     x_func = @(t) y_func(t).*x_slopes(cur_voxel);
%     z_func = @(t) y_func(t).*z_slopes(cur_voxel);
%     rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);
%     
%     rho_func_2 = @(t) sqrt(z_func(t).^2 + y_func(t).^2);
%     
%     %%
%     full_func = @(t) (norm_func(t).*laser(rho_func(t),z_func(t),t)).^2.*(1-beta.^2.*cos(2.*pi.*(z_func(t)-c.*t)/lambda).^2.*cos(theta).^2) + (norm_func(t).*laser(rho_func_2(t),x_func(t),t)).^2.*(1-beta.^2.*cos(2.*pi.*(x_func(t)-c.*t)/lambda).^2.*cos(theta).^2); %
%     
%     calc = integral(@(t) full_func(t), -integral_bound, integral_bound, 'Waypoints', t_bounds);
%     voxel_grid_phase_data_unpacked_gpu(cur_voxel) = (~isnan(calc))*calc;
%     %             if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
%     %                 voxel_grid_phase_data(m,n,cur_slice)
%     %             end
% %     waitbar(cur_voxel./num_voxels,h2,'Calculating phase shift for each voxel');
%     
% end

end
bound_increment = (max(gauss_limit*z0/c, t_limit) - min(gauss_limit*z0/c, t_limit))/10;

integral_bound = min(gauss_limit*z0/c, t_limit);

pixel_test = round(voxel_granularity/2);

cur_max = 0;

test_pixels = zeros(1,voxel_granularity);

parfor slice_inc = 1:voxel_granularity
    y_func = @(t) voxel_grid_y_data(pixel_test,pixel_test,slice_inc)-vel.*t;
    x_func = @(t) y_func(t).*voxel_grid_slope_x_data(pixel_test,pixel_test);
    z_func = @(t) y_func(t).*voxel_grid_slope_z_data(pixel_test,pixel_test);
    rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);

    %%
    norm_factor = @(t) interp1(t_range,norm_factor_array,t);
    full_func = @(t) hbar.*alpha.*n_pulse.*norm_factor(t).*laser(rho_func(t),z_func(t),t).*lambda./sqrt(mass_e^2.*(1+vel.^2/c.^2)); %
    
    %%

    cur_t_center = -voxel_grid_y_data(1,1,slice_inc)/vel;
    integral_waypoints = [-abs(voxel_grid_y_data(1,1,slice_inc))/vel,0,abs(voxel_grid_y_data(1,1,slice_inc))/vel];

    test_pixels(slice_inc) = integral(@(t) full_func(t), cur_t_center-integral_bound, cur_t_center+integral_bound, 'Waypoints', integral_waypoints);
end

[max_slice, slice_test] = max(test_pixels);

test_grid_bound_1 = zeros(voxel_granularity,voxel_granularity);
test_grid_bound_2 = zeros(voxel_granularity,voxel_granularity);
test_grid = zeros(voxel_granularity,voxel_granularity);

for m = 1:voxel_granularity
    parfor n = 1:voxel_granularity

        % calculate path for current x(t), y(t), z(t) for specific slice, and
        % voxel m,n. This is the path of the electron, but these values
        % are placed into the laser equation.

        y_func = @(t) voxel_grid_y_data(m,n,slice_test)-vel.*t;
        x_func = @(t) y_func(t).*voxel_grid_slope_x_data(m,n);
        z_func = @(t) y_func(t).*voxel_grid_slope_z_data(m,n);
        rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);

        %%
        norm_factor = @(t) interp1(t_range,norm_factor_array,t);
        full_func = @(t) hbar.*alpha.*n_pulse.*norm_factor(t).*laser(rho_func(t),z_func(t),t).*lambda./sqrt(mass_e^2.*(1+vel.^2/c.^2)); %
        
        %%
        integral_waypoints = [-abs(voxel_grid_y_data(1,1,slice_test))/vel,0,abs(voxel_grid_y_data(1,1,slice_test))/vel];

        cur_t_center = -voxel_grid_y_data(1,1,slice_test)/vel;
        test_grid_bound_1(m,n) = integral(@(t) full_func(t), cur_t_center-integral_bound, cur_t_center+integral_bound, 'Waypoints', integral_waypoints);
        %             if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
        %                 voxel_grid_phase_data(m,n,cur_slice)
        %             end
    end
end

% figure;
% subplot(1,2,1);
% imagesc(test_grid_bound_1);

integral_bound = max(gauss_limit*z0/c, t_limit);

for m = 1:voxel_granularity
    parfor n = 1:voxel_granularity

        % calculate path for current x(t), y(t), z(t) for specific slice, and
        % voxel m,n. This is the path of the electron, but these values
        % are placed into the laser equation.

        y_func = @(t) voxel_grid_y_data(m,n,slice_test)-vel.*t;
        x_func = @(t) y_func(t).*voxel_grid_slope_x_data(m,n);
        z_func = @(t) y_func(t).*voxel_grid_slope_z_data(m,n);
        rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);

        %%
        norm_factor = @(t) interp1(t_range,norm_factor_array,t);
        full_func = @(t) hbar.*alpha.*n_pulse.*norm_factor(t).*laser(rho_func(t),z_func(t),t).*lambda./sqrt(mass_e^2.*(1+vel.^2/c.^2)); %
        
        %%

        cur_t_center = -voxel_grid_y_data(1,1,slice_test)/vel;
        integral_waypoints = [-abs(voxel_grid_y_data(1,1,slice_test))/vel,0,abs(voxel_grid_y_data(1,1,slice_test))/vel];

        test_grid_bound_2(m,n) = integral(@(t) full_func(t), cur_t_center-integral_bound, cur_t_center+integral_bound, 'Waypoints', integral_waypoints);
        %             if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
        %                 voxel_grid_phase_data(m,n,cur_slice)
        %             end
    end
end

% subplot(1,2,2);
% imagesc(test_grid_bound_2);

grid_1_sum = sum(sum(test_grid_bound_1));
grid_2_sum = sum(sum(test_grid_bound_2));

prev_sum = grid_1_sum;
cur_sum = grid_2_sum;
max_sum = max([prev_sum, cur_sum]);
integral_bound = bound_increment;

% figure;

count = 1;

cur_sum_acceptable = 1;

while ((abs(cur_sum - prev_sum)/prev_sum > 0.01) && cur_sum_acceptable == 1)
    prev_sum = cur_sum;
    integral_bound = integral_bound + bound_increment;
    for m = 1:voxel_granularity
        parfor n = 1:voxel_granularity

            % calculate path for current x(t), y(t), z(t) for specific slice, and
            % voxel m,n. This is the path of the electron, but these values
            % are placed into the laser equation.

            y_func = @(t) voxel_grid_y_data(m,n,slice_test)-vel.*t;
            x_func = @(t) y_func(t).*voxel_grid_slope_x_data(m,n);
            z_func = @(t) y_func(t).*voxel_grid_slope_z_data(m,n);
            rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);

            %%
            norm_factor = @(t) interp1(t_range,norm_factor_array,t);
            full_func = @(t) hbar.*alpha.*n_pulse.*norm_factor(t).*laser(rho_func(t),z_func(t),t).*lambda./sqrt(mass_e^2.*(1+vel.^2/c.^2)); %
            
            %%
            cur_t_center = -voxel_grid_y_data(1,1,slice_test)/vel;
            integral_waypoints = [-abs(voxel_grid_y_data(1,1,slice_test))/vel,0,abs(voxel_grid_y_data(1,1,slice_test))/vel];

            test_grid(m,n) = integral(@(t) full_func(t), cur_t_center - integral_bound, cur_t_center + integral_bound, 'Waypoints', integral_waypoints);
            %             if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
            %                 voxel_grid_phase_data(m,n,cur_slice)
            %             end
        end
    end

%     subplot(10,10,count);
%     imagesc(test_grid);

    count = count + 1;
    cur_sum = sum(sum(test_grid));
    if (cur_sum >= prev_sum)
        cur_sum_acceptable = 1;
    else
        cur_sum_acceptable = 0;
    end
end

if (cur_sum_acceptable == 0)
    integral_bound = integral_bound - bound_increment; %min(gauss_limit*z0/c, t_limit);
end

integral_bound
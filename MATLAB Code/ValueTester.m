voxel_granularity = 10;
gauss_limit = 3;

planck = 6.626e-34*1e9*1e12; %nJ.*ps
hbar = planck./2./pi;
alpha = 1./137; % fine structure constant
mass_e = 9.11e-31*1e15; % pg

%% Construct laser beam
c = 2.9979e8*1e9*1e-12; % nm./ps
lambda = 500; % nm
w0 = 100e-6*1e9; % nm
sig_las = 1; % ps 
z0 = pi.*w0.^2./lambda; % Rayleigh range, nm

temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
spatial_gauss = @(rho_xy,z,t) 1./pi./omeg_las_sq(z).*exp(-(2*rho_xy.^2)./(omeg_las_sq(z).*temporal_gauss(z,t)));

laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, Inf, -gauss_limit*sig_las + c*t, gauss_limit*sig_las + c*t);

E_photon = planck.*c./lambda; % nJ
E_pulse = 1; % nJ
n_pulse = E_pulse./E_photon; % number of photons per pulse

xover_slope = 3e-3./300e-3; % 3 mm in 300 mm of travel
xover_angle = atand(xover_slope); % degrees
vel = 2.33e8*1e9/1e12; % velocity of electron, 200 kV, nm./ps
sig_ebeam = 1; % time resolution of ebeam, ps
omeg_ebeam = @(y) abs(xover_slope.*y); % nm
% e_beam = @(rho_xz,y,t) 1./pi./(omeg_ebeam(y)).^2./sig_ebeam.^2./vel.^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2).*exp(-(2.*(y-vel.*t).^2)./(sig_ebeam.^2.*vel.^2));
e_beam_xz = @(rho_xz,y) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2);
e_beam_xz_raster = @(x,y,z) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*(x.^2 + z.^2))./(omeg_ebeam(y)).^2);
e_beam_yt = @(y) 1./pi./sig_ebeam.^2./vel.^2.*exp(-(2.*(y).^2)./(sig_ebeam.^2.*vel.^2));

e_beam_yt_norm = 1./integral(e_beam_yt, -Inf, Inf);
voxel_y_weights = zeros(voxel_granularity, 1);
y_spacing = 2.*3.*sig_ebeam.*vel./voxel_granularity;
y_bounds = linspace(-3.*sig_ebeam.*vel - y_spacing./2,3.*sig_ebeam.*vel + y_spacing./2,voxel_granularity+1);

for l = 1:voxel_granularity
    voxel_y_weights(l) = e_beam_yt_norm.*integral(e_beam_yt, y_bounds(l), y_bounds(l+1));
end

cur_slice = round(voxel_granularity/2)+1;
t_spacing = min(sig_ebeam,sig_las)/voxel_granularity;
t_limit = gauss_limit*sig_ebeam;
t_range = -t_limit:t_spacing:t_limit;
laser_sum_array = zeros(size(t_range));
parfor i = 1:length(t_range)
laser_sum_array(i) = laser_sum(t_range(i));
end

norm_factor_array = 1./((0 == laser_sum_array).*1e308 + laser_sum_array);
norm_factor = @(t) interp1(t_range,norm_factor_array,t);

t2 = -z0/c:z0/c/voxel_granularity:z0/c;

t3 = -sig_ebeam:min(sig_ebeam,sig_las)/10:sig_ebeam;
z_range = -50*z0:z0/10:50*z0;
rho_range = -50*w0:w0/10:50*w0;
% figure;
% for i = 1:length(t3)
%     subplot(1,2,1);
%     plot(z_range, laser(0,z_range,t3(i)));
%     hold on;
%     subplot(1,2,2);
%     plot(rho_range, laser(rho_range,0,t3(i)));
%     hold on;
%     pause(0.5);
% end

voxel_grid_phase_data = zeros(voxel_granularity,voxel_granularity,voxel_granularity);
voxel_grid_slope_x_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_slope_z_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_y_data = zeros(voxel_granularity,voxel_granularity,voxel_granularity);

y_dist_from_center = 3.*sig_ebeam.*vel;
y_spacing = 2.*3.*sig_ebeam.*vel./voxel_granularity;

y_range = zeros(1,voxel_granularity);

for l = 1:voxel_granularity
    voxel_grid_y_data(:,:,l) = ones(voxel_granularity,voxel_granularity)*(-3.*sig_ebeam.*vel + y_spacing.*(l-1));
    y_range(l) = -3.*sig_ebeam.*vel + y_spacing.*(l-1);
end

% plot(y_range);
% pause(5);

for j = 1:voxel_granularity
    voxel_grid_slope_x_data(j,:) = linspace(-12.*omeg_ebeam(3.*sig_ebeam.*vel),12.*omeg_ebeam(3.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
    voxel_grid_slope_z_data(:,j) = linspace(-12.*omeg_ebeam(3.*sig_ebeam.*vel),12.*omeg_ebeam(3.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
end

integral_bound = min(gauss_limit*z0/c, t_limit);

h2 = figure;
for m = 1:voxel_granularity
    parfor n = 1:voxel_granularity
        
        % calculate path for current x(t), y(t), z(t) for specific slice, and
        % voxel m,n. This is the path of the electron, but these values
        % are placed into the laser equation.
        
        %% Something going wrong here to cause laser to be singular --> Smaller resolution leads to higher number of values in pop zone
        
        y_func = @(t) voxel_grid_y_data(m,n,cur_slice)+vel.*t;
        x_func = @(t) y_func(t).*voxel_grid_slope_x_data(m,n);
        z_func = @(t) y_func(t).*voxel_grid_slope_z_data(m,n);
        rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);
        
        full_func = @(t) hbar.*alpha.*n_pulse.*norm_factor(t).*laser(rho_func(t),z_func(t),t).*lambda./sqrt(mass_e^2.*(1+vel.^2/c.^2)); %

        %%%%% INCLUDE SHIFTS IN INTEGRATION BOUNDS AND UNIT ADJUSTMENTS
%         subplot(3,3,[1 4 7]);
%         plot(t2,norm_factor(t2));
%         subplot(3,3,[2 5 8]);
%         plot(t2,laser(rho_func(t2),z_func(t2),t2));
%         subplot(3,3,3);
%         plot(t2,y_func(t2));
%         subplot(3,3,6);
%         plot(t2,x_func(t2));
%         subplot(3,3,9);
%         plot(t2,z_func(t2));
%         pause(0.1);
        %
        
%         voxel_grid_phase_data(m,n,cur_slice) = integral(@(t) full_func(t), -Inf, Inf);
        voxel_grid_phase_data(m,n,cur_slice) = integral(@(t) full_func(t), -integral_bound, integral_bound);
        if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
            voxel_grid_phase_data(m,n,cur_slice)
        end
    end
end

close(h2);

% generate map distribution of electron beam, summing and averaging over
% all slices

final_phase_data = zeros(voxel_granularity, voxel_granularity);
%stdev_phase_data = zeros(voxel_granularity, voxel_granularity);

assignin('base','final_phase_data',final_phase_data);
assignin('base','voxel_grid_phase_data',voxel_grid_phase_data);
assignin('base','voxel_y_weights',voxel_y_weights);

for m = 1:voxel_granularity
    for n = 1:voxel_granularity
        for p = 1:voxel_granularity
            final_phase_data(m,n) = final_phase_data(m,n) + voxel_grid_phase_data(m,n,p).*voxel_y_weights(p);
        end
    end
end
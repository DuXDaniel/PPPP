fix(clock)

tic;

disp('Seeding workspace with relevant information.');

spmd
    warning('off','all');
end

voxel_granularity = voxgran;
slice_granularity = slicegran;
gauss_limit = 5;

planck = 6.626e-34; %J*s
hbar = planck./2./pi;
alpha = 1./137; % fine structure constant
mass_e = 9.11e-31; % kg
theta =  0;
e = 1.602e-19; % Coulombs

%% Construct laser beam
c = 2.9979e8*1e9*1e-12; % m/s
artificial_factor = 1;
lambda = 500; % nm
w0 = beam_waist; % nm
sig_las = laser_res; % ps
z0 = pi.*w0.^2./lambda; % Rayleigh range, nm
A0 = prefactor;

temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
spatial_gauss = @(rho_xy,z,t) A0./omeg_las_sq(z).*exp(-(2*rho_xy.^2)./(omeg_las_sq(z)));

laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);

laser_x_dir = @(rho_zy, x, t) spatial_gauss(rho_zy,x,t).*temporal_gauss(x,t);

laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), -gauss_limit*sig_las + c*t, gauss_limit*sig_las + c*t);



sig_ratios = [-gauss_limit -1.6449 -1.2816 -1.0364 -0.8416 -0.6745 -0.5244 -0.3853 -0.2533 -0.1257 -0.0627]; % limit, 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
weights = [0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025];

if (sig_ebeam < sig_las)
    t_bounds_overall = cell(1,length(sig_ratios)+1);
    sig_ratios = [-gauss_limit*sig_las/sig_ebeam sig_ratios];
    weights = [0.005 weights];
    num_voxels = 0;
    for i = 1:length(t_bounds_overall)-1
        t_bounds_add = linspace(sig_ratios(i)*sig_ebeam,sig_ratios(i+1)*sig_ebeam,round(weights(i)*slice_granularity));
        t_bounds_overall{i} = t_bounds_add(1:end-1);
        num_voxels = num_voxels + length(t_bounds_overall{i});
    end
    
    voxels_left = slice_granularity + 1 - num_voxels*2;
    t_bounds_overall{end} = linspace(sig_ratios(end)*sig_ebeam,-sig_ratios(end)*sig_ebeam,voxels_left);
elseif (sig_las < sig_ebeam)
    t_bounds_overall = cell(1,length(sig_ratios)+1);
    sig_ratios = [-gauss_limit*sig_ebeam/sig_las sig_ratios];
    weights = [0.005 weights];
    num_voxels = 0;
    for i = 1:length(t_bounds_overall)-1
        t_bounds_add = linspace(sig_ratios(i)*sig_las,sig_ratios(i+1)*sig_las,round(weights(i)*slice_granularity));
        t_bounds_overall{i} = t_bounds_add(1:end-1);
        num_voxels = num_voxels + length(t_bounds_overall{i});
    end
    
    voxels_left = slice_granularity + 1 - num_voxels*2;
    t_bounds_overall{end} = linspace(sig_ratios(end)*sig_las,-sig_ratios(end)*sig_las,voxels_left);
else
    t_bounds_overall = cell(1,length(sig_ratios));
    num_voxels = 0;
    for i = 1:length(t_bounds_overall)-1
        t_bounds_add = linspace(sig_ratios(i)*sig_ebeam,sig_ratios(i+1)*sig_ebeam,round(weights(i)*slice_granularity));
        t_bounds_overall{i} = t_bounds_add(1:end-1);
        num_voxels = num_voxels + length(t_bounds_overall{i});
    end
    
    voxels_left = slice_granularity + 1 - num_voxels*2;
    t_bounds_overall{end} = linspace(sig_ratios(end)*sig_ebeam,-sig_ratios(end)*sig_ebeam,voxels_left);
end

t_bounds = [];
for i = 1:length(t_bounds_overall)
    t_bounds = [t_bounds t_bounds_overall{i}];
end

for i = length(t_bounds_overall)-1:-1:1
    t_bounds = [t_bounds fliplr(-t_bounds_overall{i})];
end

t_bounds
y_bounds = t_bounds.*vel;
%y_bounds = linspace(-3.*sig_ebeam.*vel - y_spacing./2,3.*sig_ebeam.*vel + y_spacing./2,voxel_granularity+1);

for l = 1:slice_granularity
    voxel_y_weights(l) = e_beam_yt_norm.*integral(e_beam_yt, y_bounds(l), y_bounds(l+1));
end

% voxel_y_weights = xlsread('Y_Weights.xlsx');
t_range = t_bounds;
laser_sum_array = zeros(size(t_range));
parfor i = 1:length(t_range)
laser_sum_array(i) = laser_sum(t_range(i));
end

norm_factor_array = A0./((0 == laser_sum_array).*1e308 + laser_sum_array);



%% Goal: Calculate how much energy passes through given area around xz over a period of time
% Triple integral (first in an area dictated by rho_xy at z = 0 at some t, then over some t range).
% This will give the total energy in that range of time.



toc;
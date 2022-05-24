function [voxel_grid_phase_data] = Quasiclassical_Func_Double_Laser_V01_SM_trapz(beam_waist, laser_res, e_res, prefactor)

tic;

disp('Seeding workspace with relevant information.');

spmd
    warning('off','all');
end

voxel_granularity = 81;
slice_granularity = 121;
gauss_limit = 3;

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

% --> Loop structure, cannot actually store the voxels in RAM like I wish I
% could, (10k x 10k x 10k = 8 TB RAM) so have to do it the slow way and
% advance each slice individually while calculating the full time range for
% each slice. Forget each slice at the end of each time range and move to
% the next.

% Will note, can do 1k x 1k x 1k fairly easily. Shift to lower resolution?
% seems reasonable, really only trying to get an idea of the phase profile.

%%%%% ALIGNMENT SYMMETRY?
% Could cut down arrays by 1./2 if accounting for the fact that there is
% phase symmetry in the way the beams move (i.e. only need to calculate up
% to the peak before simply applying rotational symmetry (mirrored upwards,
% rotated 180) --> Add this later (no need to get into complexities now)

%% Construct electron beam
% Each voxel in the electron packet will contain some weighted number,
% totaling to 1 when all of them are summed.

xover_slope = 3e-3./300e-3; % 3 mm in 300 mm of travel
xover_angle = atand(xover_slope); % degrees
vel = 2.33e8*1e9/1e12; % velocity of electron, 200 kV, nm./ps
beta = vel/c;
gamma = (1-beta^2)^(-1/2);
sig_ebeam = e_res; % time resolution of ebeam, ps
omeg_ebeam = @(y) abs(xover_slope.*y); % nm
% e_beam = @(rho_xz,y,t) 1./pi./(omeg_ebeam(y)).^2./sig_ebeam.^2./vel.^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2).*exp(-(2.*(y-vel.*t).^2)./(sig_ebeam.^2.*vel.^2));
e_beam_xz = @(rho_xz,y) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2);
e_beam_xz_raster = @(x,y,z) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*(x.^2 + z.^2))./(omeg_ebeam(y)).^2);
e_beam_yt = @(y) 1./pi./sig_ebeam.^2./vel.^2.*exp(-(2.*(y).^2)./(sig_ebeam.^2.*vel.^2));

t_limit = gauss_limit*sig_ebeam;

% --> Key: don't need to calculate instantaneous density of electrons, just
% need the path and the ending normalization.

%%%% Construct voxels
% how granular do I want the voxels if they all pass through (0,0) at
% crossover? It's arbitrary at that point. Let us assume a 1000 x 1000
% grid split across 6-sigma centered at the peak

e_beam_xz_norm = 1./integral(@(x) e_beam_xz(x,3.*sig_ebeam.*vel), 0, Inf);

%%%% Construct slices
% Do initial normalization, then segment into the slices slices dependent
% on e_pulse res (let's assume 1000 slices?) or perhaps make it such that
% it is the smaller of the laser or e pulse resolutions divided by 10000 as
% the spacing?

%%%% not really necessary if viewing the voxels on a square grid (weights
%%%% only need to be applied in vertical (y) direction).

% xz_spacing = 2.*(omeg_ebeam(3.*sig_ebeam.*vel))./voxel_granularity;
% xz_bounds = linspace(-omeg_ebeam(3.*sig_ebeam.*vel)-xz_spacing./2, omeg_ebeam(3.*sig_ebeam.*vel)+xz_spacing./2, voxel_granularity+1);
% 
% h = waitbar(0,'Populating xz Grid Weights');
% for j = 1:voxel_granularity
%     for k = 1:voxel_granularity
%         voxel_xz_grid_weights(j,k) = e_beam_xz_norm.*integral2(@(x,y) e_beam_xz_raster(x,3.*sig_ebeam.*vel,y), xz_bounds(j), xz_bounds(j+1), xz_bounds(k), xz_bounds(k+1));
%     end
%     waitbar(j./voxel_granularity);
% end
% close(h);

% voxel_xz_grid_weights = xlsread('XZ_Weights.xlsx');

sig_ratios = [-gauss_limit -1.6449 -1.2816 -1.0364 -0.8416 -0.6745 -0.5244 -0.3853 -0.2533 -0.1257 -0.0627]; % limit, 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
weights = [0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025];

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
    weights(1) = 0.075;
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

e_beam_yt_norm = 1./integral(e_beam_yt, y_bounds(1), y_bounds(end));

% --> Normalize on a per slice basis here
% --> which is to say, ignore the spatial component and normalize the
% temporal./y component and attach each slice a portion of that
% normalization
% --> Then simply multiply that slice's normalization factor to the
% individual voxel of each slice at the maximum expansion point after
% Compton scattering. Just take the distribution at t_end.

%% calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
% voxel_xz_grid_weights = zeros(voxel_granularity, voxel_granularity);
voxel_y_weights = zeros(slice_granularity, 1);

for l = 1:slice_granularity
    voxel_y_weights(l) = e_beam_yt_norm.*integral(e_beam_yt, y_bounds(l), y_bounds(l+1));
end

sum(voxel_y_weights)

% voxel_y_weights = xlsread('Y_Weights.xlsx');
t_range = t_bounds;
laser_sum_array = zeros(size(t_range));
parfor i = 1:length(t_range)
laser_sum_array(i) = laser_sum(t_range(i));
end

norm_factor_array = A0./((0 == laser_sum_array).*1e308 + laser_sum_array);

norm_spline = spline(t_range,norm_factor_array);
norm_func = @(t) ppval(norm_spline,t);

%%%% Construct pathway of travel for all voxels
% pathway of travel is identical at all positions of xz, so just find a
% generic pathway of travel for a double cone base the pathway off of the
% crossover angle (just define by slope maybe?) --> Seems like another
% variable to take note of

% Need to calculate changing width of e-beam

%%%% voxel square sizes can be same because of travel path. Just change
% magnitude of Gaussian from back focal plane

% How to populate location data? Want the lower half of the e-beam
% (symmetry) and 3-sigma from the center
% can populate at time zero and then let integral back-calculate from -Inf.
voxel_grid_phase_data = zeros(voxel_granularity,voxel_granularity,slice_granularity);
voxel_grid_slope_x_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_slope_z_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_y_data = zeros(voxel_granularity,voxel_granularity,slice_granularity);

y_range = zeros(1,slice_granularity);
t_range = zeros(1,slice_granularity);

for l = 1:slice_granularity
    voxel_grid_y_data(:,:,l) = ones(voxel_granularity,voxel_granularity)*((y_bounds(l)+y_bounds(l+1))/2);
    y_range(l) = (y_bounds(l)+y_bounds(l+1))/2;
    t_range(l) = (t_bounds(l)+t_bounds(l+1))/2;
end

%min_t_vals = max([200, slice_granularity]);
%if (min_t_vals ~= slice_granularity)
min_t_vals_sep = 2000;%round(min_t_vals/slice_granularity) + slice_granularity;
num_points_add = @(t) -1./(1+exp(-abs(t)/(5*t_range(end))))+1;
add_points = round(min_t_vals_sep*num_points_add(t_range));
tot_points = sum(add_points);
t_range_extended = zeros(1,tot_points - add_points(end));
tot_added = 0;
for l = 1:slice_granularity-1
    t_range_extended(tot_added + 1:tot_added + add_points(l) + 1) = linspace(t_range(l),t_range(l+1),add_points(l)+1);
    tot_added = tot_added + add_points(l);
end
assignin('base','t_range_extended',t_range_extended);
%else
%    t_range_extended = t_range;
%end

if (sig_ebeam < sig_las)
    y_dist_from_center = gauss_limit.*sig_las.*vel;
    for j = 1:voxel_granularity
        voxel_grid_slope_x_data(j,:) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_las.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_las.*vel),voxel_granularity)./y_dist_from_center;
        voxel_grid_slope_z_data(:,j) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_las.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_las.*vel),voxel_granularity)./y_dist_from_center;
    end
elseif (sig_las <= sig_ebeam)
    y_dist_from_center = gauss_limit.*sig_ebeam.*vel;
    for j = 1:voxel_granularity
        voxel_grid_slope_x_data(j,:) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
        voxel_grid_slope_z_data(:,j) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
    end
end

%% Loop
tic;

integral_bound = t_range(end);

num_voxels = slice_granularity*voxel_granularity^2;

init_y_vals = zeros(num_voxels,1);
x_slopes = zeros(num_voxels,1);
z_slopes = zeros(num_voxels,1);
for cur_voxel = 0:num_voxels-1
    cur_slice = floor(cur_voxel/voxel_granularity^2)+1;
    cur_voxel_num = mod(cur_voxel,voxel_granularity^2);
    m = floor(cur_voxel_num/voxel_granularity)+1;
    n = mod(cur_voxel_num,voxel_granularity)+1;
    
    init_y_vals(cur_voxel+1) = voxel_grid_y_data(m,n,cur_slice);
    x_slopes(cur_voxel+1) = voxel_grid_slope_x_data(m,n);
    z_slopes(cur_voxel+1) = voxel_grid_slope_z_data(m,n);
end

voxel_grid_phase_data_unpacked = quasiclassical_trapz_double_calculation(num_voxels,init_y_vals,x_slopes,z_slopes,t_range_extended,norm_factor_array,laser,norm_func,beta,c,lambda,theta,integral_bound,t_bounds,vel);

for cur_voxel = 0:num_voxels-1
    cur_slice = floor(cur_voxel/voxel_granularity^2)+1;
    cur_voxel_num = mod(cur_voxel,voxel_granularity^2);
    m = floor(cur_voxel_num/voxel_granularity)+1;
    n = mod(cur_voxel_num,voxel_granularity)+1;
    voxel_grid_phase_data(m,n,cur_slice) = voxel_grid_phase_data_unpacked(cur_voxel+1);
end
toc;

% generate map distribution of electron beam, summing and averaging over
% all slices

final_phase_data = zeros(voxel_granularity, voxel_granularity);
%stdev_phase_data = zeros(voxel_granularity, voxel_granularity);

% assignin('base','final_phase_data',final_phase_data);
% assignin('base','voxel_grid_phase_data',voxel_grid_phase_data);
% assignin('base','voxel_y_weights',voxel_y_weights);

for m = 1:voxel_granularity
    for n = 1:voxel_granularity
        for p = 1:slice_granularity
            final_phase_data(m,n) = final_phase_data(m,n) + voxel_grid_phase_data(m,n,p).*voxel_y_weights(p);
        end
    end
end

end
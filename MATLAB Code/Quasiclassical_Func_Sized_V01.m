function [final_phvoxel_grid_cumulative_phase_dataase_data] = Quasiclassical_Func_Sized_V01(beam_waist, laser_res, e_res, prefactor)

tic;

disp('Seeding workspace with relevant information.');

spmd
    warning('off','all');
end

voxel_granularity = 81;
slice_granularity = 81;
focus_granularity = 31;
gauss_limit = 3;

size_direct_beam = 100e3; % nm

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
e_beam_xz = @(rho_xz) 1./pi./(size_direct_beam).^2.*exp(-(2.*rho_xz.^2)./(size_direct_beam).^2);
e_beam_xz_raster = @(x,z) 1./pi./(size_direct_beam).^2.*exp(-(2.*(x.^2 + z.^2))./(size_direct_beam).^2);
e_beam_yt = @(y) 1./pi./sig_ebeam.^2./vel.^2.*exp(-(2.*(y).^2)./(sig_ebeam.^2.*vel.^2));

t_limit = gauss_limit*sig_ebeam;

% --> Key: don't need to calculate instantaneous density of electrons, just
% need the path and the ending normalization.

%%%% Construct voxels
% how granular do I want the voxels if they all pass through (0,0) at
% crossover? It's arbitrary at that point. Let us assume a 1000 x 1000
% grid split across 6-sigma centered at the peak

e_beam_xz_norm = 1./integral(@(x)2.*pi.*x.*e_beam_xz(x), 0, Inf);

%%%% Construct slices
% Do initial normalization, then segment into the slices slices dependent
% on e_pulse res (let's assume 1000 slices?) or perhaps make it such that
% it is the smaller of the laser or e pulse resolutions divided by 10000 as
% the spacing?

e_beam_yt_norm = 1./integral(e_beam_yt, -Inf, Inf);

% --> Normalize on a per slice basis here
% --> which is to say, ignore the spatial component and normalize the
% temporal./y component and attach each slice a portion of that
% normalization
% --> Then simply multiply that slice's normalization factor to the
% individual voxel of each slice at the maximum expansion point after
% Compton scattering. Just take the distribution at t_end.

%% calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
% calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
voxel_xz_grid_weights = zeros(voxel_granularity, voxel_granularity);
voxel_y_weights = zeros(slice_granularity, 1);

%%%% not really necessary if viewing the voxels on a square grid (weights
%%%% only need to be applied in vertical (y) direction).

xz_spacing = 2.*(size_direct_beam)./voxel_granularity;
xz_centers = linspace(-size_direct_beam-xz_spacing./2, size_direct_beam+xz_spacing./2, voxel_granularity);
xz_bounds = linspace(-size_direct_beam-xz_spacing./2, size_direct_beam+xz_spacing./2, voxel_granularity+1);
xz_focus_centers = linspace(-size_direct_beam-xz_spacing./2, size_direct_beam+xz_spacing./2, focus_granularity);
xz_focus_bounds = linspace(-size_direct_beam-xz_spacing./2, size_direct_beam+xz_spacing./2, focus_granularity+1);

h = waitbar(0,'Populating xz Grid Weights');
for j = 1:voxel_granularity
    parfor k = 1:voxel_granularity
        voxel_xz_grid_weights(j,k) = e_beam_xz_norm.*integral2(@(x,z) e_beam_xz_raster(x,z), xz_bounds(j), xz_bounds(j+1), xz_bounds(k), xz_bounds(k+1));
    end
    waitbar(j./voxel_granularity);
end
close(h);

% voxel_xz_grid_weights = xlsread('XZ_Weights.xlsx');

y_spacing = 2.*3.*sig_ebeam.*vel./slice_granularity;
y_bounds = linspace(-3.*sig_ebeam.*vel - y_spacing./2,3.*sig_ebeam.*vel + y_spacing./2,slice_granularity+1);

for l = 1:slice_granularity
    voxel_y_weights(l) = e_beam_yt_norm.*integral(e_beam_yt, y_bounds(l), y_bounds(l+1));
end

% voxel_y_weights = xlsread('Y_Weights.xlsx');

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
voxel_grid_cumulative_phase_data = zeros(focus_granularity,focus_granularity);
curwl_matrix = zeros(focus_granularity,focus_granularity);
curwh_matrix = zeros(focus_granularity,focus_granularity);
curql_matrix = zeros(focus_granularity,focus_granularity);
curqh_matrix = zeros(focus_granularity,focus_granularity);
voxel_grid_phase_data = zeros(voxel_granularity,voxel_granularity,slice_granularity);
voxel_grid_slope_x_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_slope_z_data = zeros(voxel_granularity,voxel_granularity); % travel path information for each voxel, only need one representative plane for each
voxel_grid_y_data = zeros(voxel_granularity,voxel_granularity,slice_granularity);

y_dist_from_center = gauss_limit.*sig_ebeam.*vel;
y_spacing = 2.*gauss_limit.*sig_ebeam.*vel./slice_granularity;

y_range = zeros(1,slice_granularity);

for l = 1:slice_granularity
    voxel_grid_y_data(:,:,l) = ones(voxel_granularity,voxel_granularity)*(-gauss_limit.*sig_ebeam.*vel + y_spacing.*(l-1));
    y_range(l) = -gauss_limit.*sig_ebeam.*vel + y_spacing.*(l-1);
end

% plot(y_range);
% pause(5);

for j = 1:voxel_granularity
    voxel_grid_slope_x_data(j,:) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
    voxel_grid_slope_z_data(:,j) = linspace(-gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),gauss_limit.*omeg_ebeam(gauss_limit.*sig_ebeam.*vel),voxel_granularity)./y_dist_from_center;
end

for u = 1:focus_granularity
    for v = 1:focus_granularity
        
        curu1 = xz_focus_bounds(u);
        curu2 = xz_focus_bounds(u+1);
        curv1 = xz_focus_bounds(v);
        curv2 = xz_focus_bounds(v+1);
        
        curwh = voxel_granularity;
        foundwh = 0;
        curwl = 1;
        foundwl = 0;
        curqh = voxel_granularity;
        foundqh = 0;
        curql = 1;
        foundql = 0;
        for q = 1:length(xz_centers)
            if xz_centers(q) <= curu2 && foundwh ~= 1 && foundwl == 1
                curwh = q;
                foundwh = 1;
            elseif xz_centers(q) >= curu1 && foundwl ~= 1
                foundwl = 1;
                curwl = q;
            end
            
            if xz_centers(q) <= curv2 && foundqh ~= 1 && foundql == 1
                curqh = q;
                foundqh = 1;
            elseif xz_centers(q) >= curv1 && foundql ~= 1
                foundql = 1;
                curql = q;
            end
        end
        
        curwl_matrix(u,v) = curwl;
        curwh_matrix(u,v) = curwh;
        curql_matrix(u,v) = curql;
        curqh_matrix(u,v) = curqh;
        
    end
end

voxel_xz_focus_weights = zeros(focus_granularity,focus_granularity);
num_condense = zeros(focus_granularity,focus_granularity);
for u = 1:focus_granularity
    for v = 1:focus_granularity
        num_condense(u,v) = abs((curwh_matrix(u,v)-curwl_matrix(u,v)+1)*(curqh_matrix(u,v)-curql_matrix(u,v)+1));
        voxel_xz_focus_weights(u,v) = sum(sum(voxel_xz_grid_weights(curwl_matrix(u,v):curwh_matrix(u,v),curql_matrix(u,v):curqh_matrix(u,v))));
    end
end

integral_bound = min(gauss_limit*z0/c, t_limit);

z_drift = linspace(-100e3,100e3,focus_granularity);
x_drift = linspace(-100e3,100e3,focus_granularity);

toc;

%% Loop

h2 = waitbar(0, 'Calculating phase shift for each slice');

for zin = 1:length(z_drift)
    
    for xin = 1:length(x_drift)
        
        for cur_slice = 1:slice_granularity
            
            tic;
            
            % Determine slice level --> will determine weighting at the end
            
            % Assumption: all electrons pass through (x0,z0) at crossover most
            % likely incorrect, but we have nothing else to go off of will only
            % slightly cause the CTF to show a higher than normal resolution
            
            % reference current voxel xz position grid from travel path
            % calculation and current time
            
            % calculate photon densities at position grid
            for m = 1:voxel_granularity
                for n = 1:voxel_granularity
                    
                    % calculate path for current x(t), y(t), z(t) for specific slice, and
                    % voxel m,n. This is the path of the electron, but these values
                    % are placed into the laser equation.
                    
                    y_func = @(t) voxel_grid_y_data(m,n,cur_slice)-vel.*t;
                    x_func = @(t) y_func(t).*voxel_grid_slope_x_data(m,n);
                    z_func = @(t) y_func(t).*voxel_grid_slope_z_data(m,n);
                    rho_func = @(t) sqrt(x_func(t).^2 + y_func(t).^2);
                    
                    %%
                    full_func = @(t) laser(rho_func(t),z_func(t),t).^2.*(1-beta.^2.*cos(2.*pi.*(z_func(t)-c.*t)/lambda).^2.*cos(theta).^2); %
                    
                    %%
                    
                    integral_waypoints = [-abs(voxel_grid_y_data(1,1,cur_slice))/vel,0,abs(voxel_grid_y_data(1,1,cur_slice))/vel];
                    
                    calc = artificial_factor.*integral(@(t) full_func(t), -Inf, Inf, 'Waypoints', integral_waypoints);
                    voxel_grid_phase_data(m,n,cur_slice) = (~isnan(calc))*calc;
                    %             if (voxel_grid_phase_data(m,n,cur_slice) ~= 0)
                    %                 voxel_grid_phase_data(m,n,cur_slice)
                    %             end
                end
            end
            
            %     imagesc(voxel_grid_phase_data(:,:,cur_slice));
            
            toc;
        end
        
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
        
        for u = 1:focus_granularity
            for v = 1:focus_granularity
                voxel_grid_cumulative_phase_data(u,v) = voxel_grid_cumulative_phase_data(u,v) + sum(sum(final_phase_data(curwl_matrix(u,v):curwh_matrix(u,v),curql_matrix(u,v):curqh_matrix(u,v))))/num_condense(u,v)*voxel_xz_focus_weights(zin,xin);
            end
        end
        
        xin
        
    end
    
    waitbar(zin./focus_granularity,h2,'Calculating phase shift for each slice');
    zin
    
end

close(h2);

end
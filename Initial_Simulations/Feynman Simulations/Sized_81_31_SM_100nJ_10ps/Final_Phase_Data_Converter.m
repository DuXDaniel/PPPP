function [final_phase_data] = Final_Phase_Data_Converter(beam_waist, laser_res, e_res, prefactor, voxel_grid_phase_data)

tic;

disp('Seeding workspace with relevant information.');

spmd
    warning('off','all');
end

voxel_granularity = 128;
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
spatial_gauss = @(rho_xy,z,t) A0./omeg_las_sq(z).*exp(-(2*rho_xy.^2)./(omeg_las_sq(z).*temporal_gauss(z,t)));

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

e_beam_yt_norm = 1./integral(e_beam_yt, -Inf, Inf);

% --> Normalize on a per slice basis here
% --> which is to say, ignore the spatial component and normalize the
% temporal./y component and attach each slice a portion of that
% normalization
% --> Then simply multiply that slice's normalization factor to the
% individual voxel of each slice at the maximum expansion point after
% Compton scattering. Just take the distribution at t_end.

%% calculating voxel weights... need to find integration bounds for each voxel point at the final position of the e-beam ("detector")
% voxel_xz_grid_weights = zeros(voxel_granularity, voxel_granularity);
voxel_y_weights = zeros(voxel_granularity, 1);

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

y_spacing = 2.*gauss_limit.*sig_ebeam.*vel./voxel_granularity;
y_bounds = linspace(-3.*sig_ebeam.*vel - y_spacing./2,3.*sig_ebeam.*vel + y_spacing./2,voxel_granularity+1);

for l = 1:voxel_granularity
    voxel_y_weights(l) = e_beam_yt_norm.*integral(e_beam_yt, y_bounds(l), y_bounds(l+1));
end

final_phase_data = zeros(voxel_granularity, voxel_granularity);
%stdev_phase_data = zeros(voxel_granularity, voxel_granularity);

% assignin('base','final_phase_data',final_phase_data);
% assignin('base','voxel_grid_phase_data',voxel_grid_phase_data);
% assignin('base','voxel_y_weights',voxel_y_weights);

for m = 1:voxel_granularity
    for n = 1:voxel_granularity
        for p = 1:voxel_granularity
            final_phase_data(m,n) = final_phase_data(m,n) + voxel_grid_phase_data(m,n,p).*voxel_y_weights(p);
        end
    end
end

end
function [laser_sum_array] = laser_norm_testing(beam_waist, laser_res, e_res, pulse_energy)


tic;

disp('Seeding workspace with relevant information.');

spmd
    warning('off','all');
end

voxel_granularity = 121;
slice_granularity = 121;
gauss_limit = 3;

planck = 6.626e-34*1e9*1e12; %nJ.*ps
hbar = planck./2./pi;
alpha = 1./137; % fine structure constant
mass_e = 9.11e-31*1e15; % pg

%% Construct laser beam
c = 2.9979e8*1e9*1e-12; % nm./ps
lambda = 500; % nm
w0 = beam_waist; % nm
sig_las = laser_res; % ps
z0 = pi.*w0.^2./lambda; % Rayleigh range, nm

temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
spatial_gauss = @(rho_xy,z,t) 1./pi./omeg_las_sq(z).*exp(-(2*rho_xy.^2)./(omeg_las_sq(z)./temporal_gauss(z,t)));

laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), -gauss_limit*sig_las + c*t, gauss_limit*sig_las + c*t);

E_photon = planck.*c./lambda; % nJ
E_pulse = pulse_energy; % nJ
n_pulse = E_pulse./E_photon; % number of photons per pulse

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
sig_ebeam = e_res; % time resolution of ebeam, ps
omeg_ebeam = @(y) abs(xover_slope.*y); % nm
% e_beam = @(rho_xz,y,t) 1./pi./(omeg_ebeam(y)).^2./sig_ebeam.^2./vel.^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2).*exp(-(2.*(y-vel.*t).^2)./(sig_ebeam.^2.*vel.^2));
e_beam_xz = @(rho_xz,y) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*rho_xz.^2)./(omeg_ebeam(y)).^2);
e_beam_xz_raster = @(x,y,z) 1./pi./(omeg_ebeam(y)).^2.*exp(-(2.*(x.^2 + z.^2))./(omeg_ebeam(y)).^2);
e_beam_yt = @(y) 1./pi./sig_ebeam.^2./vel.^2.*exp(-(2.*(y).^2)./(sig_ebeam.^2.*vel.^2));

% --> Key: don't need to calculate instantaneous density of electrons, just
% need the path and the ending normalization.

%% establishing interpolated normalization for laser
t_spacing = min(sig_ebeam,sig_las)/slice_granularity;
t_limit = gauss_limit*sig_ebeam;
t_range = -t_limit:t_spacing:t_limit;
t_add_left = [-100*t_limit:t_spacing:-t_limit-t_spacing];
t_add_right = [t_limit+t_spacing:t_spacing:100*t_limit];
enlargened_t_range = [t_add_left t_range t_add_right];
laser_sum_array = zeros(size(t_range));
parfor i = 1:length(t_range)
laser_sum_array(i) = laser_sum(t_range(i));
end

norm_factor_array = 1./((0 == laser_sum_array).*1e308 + laser_sum_array);

end
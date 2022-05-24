gauss_limit = 5;
E_pulse = 4.5e3; % nJ
w0 = 100e3; % nm
sig_las = 10; % ps

planck = 6.626e-34*1e9*1e12; %nJ.*ps
hbar = planck./2./pi;
alpha = 1./137; % fine structure constant
mass_e = 9.11e-31*1e15; % pg

%% Construct laser beam
c = 2.9979e8*1e9*1e-12; % nm./ps
lambda = 500; % nm
z0 = pi.*w0.^2./lambda; % Rayleigh range, nm

temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
spatial_gauss = @(rho_xy,z,t) 1./pi./omeg_las_sq(z).*exp(-(2*rho_xy.^2)./(omeg_las_sq(z)));

laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
laser_sum = @(t,int_bound_ratio) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), -c*(int_bound_ratio*sig_las + t), c*(int_bound_ratio*sig_las + t));

sig_las_range = 1e-3:1e-3:gauss_limit;

laser_sum_range = zeros(size(sig_las_range));

full_laser_sum = laser_sum(0,gauss_limit);
full_norm_factor = E_pulse/full_laser_sum;

for i = 1:length(sig_las_range)
    laser_sum_range(i) = laser_sum(0,sig_las_range(i));
end

energy_sum_range = laser_sum_range*full_norm_factor/1e9; % J
power_sum_range = energy_sum_range./sig_las_range./sig_las*1e12; % W
fluence_sum_range = power_sum_range./pi./w0.^2./2*(1e7)^2; % W/cm^2
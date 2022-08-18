gauss_limit = 3;

planck = 6.626e-34*1e9*1e12; %nJ.*ps
hbar = planck./2./pi;
alpha = 1./137; % fine structure constant
mass_e = 9.11e-31*1e15; % pg

red_ramp = linspace(227/256,70/256,12);
green_ramp = linspace(180/256,91/256.75,12);
blue_ramp = linspace(70/256,227/256,12);
newcolors = [red_ramp', green_ramp', blue_ramp'];

%% Construct laser beam
c = 2.9979e8*1e9*1e-12; % nm./ps
lambda = 500; % nm
w0 = 100e3; % nm
sig_las = 10; % ps
z0 = pi.*w0.^2./lambda; % Rayleigh range, nm

temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
spatial_gauss = @(rho_xy,z,t) 1.*exp(-(2*rho_xy.^2)./(omeg_las_sq(z))); % ./pi./omeg_las_sq(z)

laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), (-gauss_limit*sig_las + t)*c, (gauss_limit*sig_las + t)*c);

z_range = linspace(-3*sig_las*c,3*sig_las*c,101);
x_range = linspace(-3*w0,3*w0,101);
y_range = linspace(-3*w0,3*w0,101);
t_range = linspace(-3*sig_las,3*sig_las,101);

laser_sum_array = zeros(size(t_range));
parfor i = 1:length(t_range)
    laser_sum_array(i) = laser_sum(t_range(i));
end

norm_factor_array = 1./((0 == laser_sum_array).*1e308 + laser_sum_array);
norm_factor = @(t) interp1(t_range,norm_factor_array,t);

laser_full = @(rho_xy,z,t) norm_factor(t).*laser(rho_xy,z,t); %

soln_full = zeros(length(z_range),length(t_range));
soln_spatial_z = zeros(length(z_range),length(t_range));
soln_spatial_x = zeros(length(x_range), length(t_range));
soln_spatial_y = zeros(length(y_range), length(t_range));
soln_temporal = zeros(length(z_range),length(t_range));

for i = 1:length(z_range)
    parfor j = 1:length(t_range)
        soln_spatial_z(i,j) = spatial_gauss(0,z_range(i),t_range(j));
        soln_temporal(i,j) = temporal_gauss(z_range(i),t_range(j));
        soln_full(i,j) = laser_full(0,z_range(i),t_range(j));
    end
end

for i = 1:length(x_range)
    parfor j = 1:length(t_range)
        soln_spatial_x(i,j) = laser_full(x_range(i),0, t_range(j));
    end
end

for i = 1:length(y_range)
    parfor j = 1:length(t_range)
        soln_spatial_y(i,j) = laser_full(y_range(i),0, t_range(j));
    end
end

zfig = figure;
xfig = figure;
yfig = figure;

for i = 1:10:length(t_range)
    curcount = (i-1)/10+1;
    figure(zfig);
    set(gca, 'YScale', 'log');
    hold on;
    semilogy(z_range./1e9,soln_full(:,i),'Color',newcolors(curcount,:));
    pbaspect([1 1 1]);
    %     subplot(3,3,2);
    %     set(gca, 'YScale', 'log');
    %     hold on;
    %     semilogy(z_range,soln_spatial_z(:,i));
    figure(xfig);
    set(gca, 'YScale', 'log');
    hold on;
    loglog(x_range./1e6,soln_spatial_x(:,i),'Color',newcolors(curcount,:));
    pbaspect([1 1 1]);
    figure(yfig);
    set(gca, 'YScale', 'log');
    hold on;
    loglog(y_range./1e6,soln_spatial_y(:,i),'Color',newcolors(curcount,:));
    pbaspect([1 1 1]);
    %     subplot(3,3,[3 6 9]);
    %     set(gca, 'YScale', 'log');
    %     hold on;
    %     semilogy(z_range,soln_temporal(:,i));
    pause(0.5);
end
figure(zfig);
box on;
set(gca,'FontSize',16);
xlabel('z-axis distance from waist (m)','FontSize',26);
ylabel('Intensity','FontSize',26);
figure(xfig);
box on;
set(gca,'FontSize',16);
xlabel('radial distance from waist (mm)','FontSize',26);
ylabel('Intensity','FontSize',26);
figure(yfig);
box on;
set(gca,'FontSize',16);
xlabel('radial distance from waist (mm)','FontSize',26);
ylabel('Intensity','FontSize',26);

sig_las_range = [1,10,100,1e3,10e3,100e3];

soln_spatial_sig_las = zeros(101,length(sig_las_range));
w0 = 100e3;

sigfig = figure;
box on;
pbaspect([2 1 1]);

for k = 1:length(sig_las_range)
    
    cur_sig_las = sig_las_range(k);
    temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(cur_sig_las.^2.*c.^2));
    omeg_las_sq = @(z) w0.^2*(1+z.^2/z0.^2);
    spatial_gauss = @(rho_xy,z,t) 1.*exp(-(2*rho_xy.^2)./(omeg_las_sq(z))); % ./pi./omeg_las_sq(z)

    laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
    laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), (-gauss_limit*sig_las + t)*c, (gauss_limit*sig_las + t)*c);
    
    z_range = linspace(-3*cur_sig_las*c,3*cur_sig_las*c,101);
    
    laser_sum_array = laser_sum(0);
    
    norm_factor_array = 1./((0 == laser_sum_array).*1e308 + laser_sum_array);
    norm_factor = @(t) norm_factor_array;
    
    laser_full = @(rho_xy,z,t) norm_factor(t).*laser(rho_xy,z,t); %
    
    for i = 1:length(z_range)
        soln_spatial_sig_las(i,k) = laser_full(0,z_range(i),0);
    end
    
    set(gca, 'YScale', 'log');
    hold on;
    plot(z_range./1e9,soln_spatial_sig_las(:,k),'Color',newcolors((k*2)-1,:));
end
set(gca,'FontSize',16);
xlabel('z-axis distance from waist (m)','FontSize',26);
ylabel('Intensity','FontSize',26);

cell_sig_las_range = cellstr(num2str(sig_las_range));
a = legend(strsplit(cell_sig_las_range{1},' '),'FontSize', 16);
title(a,'Temporal Resolution (ps)');

w0_range = [500, 1000, 10e3, 100e3, 1e6];

soln_spatial_w0 = zeros(101,length(w0_range));
sig_las = 1e3;

w0fig = figure;
box on;
pbaspect([2 1 1]);

for k = 1:length(w0_range)
    curw0 = w0_range(k);
    temporal_gauss = @(z,t) exp(-(2.*(z-c.*t).^2)./(sig_las.^2.*c.^2));
    omeg_las_sq = @(z) curw0.^2*(1+z.^2/z0.^2);
    spatial_gauss = @(rho_xy,z,t) 1.*exp(-(2*rho_xy.^2)./(omeg_las_sq(z))); % ./pi./omeg_las_sq(z)
    
    laser = @(rho_xy, z, t) spatial_gauss(rho_xy,z,t).*temporal_gauss(z,t);
    laser_sum = @(t) integral2(@(x,y) 2.*pi.*x.*laser(x,y,t), 0, gauss_limit*sqrt(omeg_las_sq(c*t)), (-gauss_limit*sig_las + t)*c, (gauss_limit*sig_las + t)*c);
    
    z_range = linspace(-3*sig_las*c,3*sig_las*c,101);
    
    laser_sum_array = laser_sum(0);
    
    norm_factor_array = 1./((0 == laser_sum_array).*1e308 + laser_sum_array);
    norm_factor = @(t) laser_sum_array;
    
    laser_full = @(rho_xy,z,t) norm_factor(t).*laser(rho_xy,z,t); %
    
    for i = 1:length(z_range)
        soln_spatial_w0(i,k) = laser_full(0,z_range(i),0);
    end
    
    set(gca, 'YScale', 'log');
    hold on;
    plot(z_range./1e9,soln_spatial_w0(:,k),'Color',newcolors((k*2),:));
end
set(gca,'FontSize',16);
xlabel('z-axis distance from waist (m)','FontSize',26);
ylabel('Intensity','FontSize',26);

cell_w0_range = cellstr(num2str(w0_range));
b = legend(strsplit(cell_w0_range{1},' '),'FontSize', 16);
title(b,'WaistSize (nm)');
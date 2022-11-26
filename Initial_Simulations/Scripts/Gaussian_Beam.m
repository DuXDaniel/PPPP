classdef Gaussian_Beam < handle
    
    properties
        % Assuming that the index of refraction is equivalent to 1
        A0; % Is variable to the position of the laser relative to the beam waist, will have to generalize the derivation from beam-waist position
        E_p;
        w0;
        lambda;
        poynting; % vector of direction of travel
        poynting_norm;
        u; % vector of positive E-field (i.e. polarization)
        waist_loc; % vector of waist location (x,y,z)
        rayleigh_range;
        time_res; % time stdev of the pulse
        norm_factor; 
        cur_time;
        normalized;
    end
    
    properties (Constant)
        c = 2.9979e8; % m/s
        h = 6.626e-34; % J*s
    end
    
    methods
        function obj = Gaussian_Beam(pulse_energy,beamWaist,wavelength,dirTravel,polarization,waistPos,temporalRes,curTime)
            obj.E_p = pulse_energy;
            obj.w0 = beamWaist;
            obj.lambda = wavelength;
            obj.poynting = dirTravel;
            obj.poynting_norm = obj.poynting./(sum(sqrt(obj.poynting.*obj.poynting)));
            obj.u = polarization;
            obj.waist_loc = waistPos;
            obj.time_res = temporalRes;
            obj.cur_time = curTime;
            obj.normalized = 0;
            
            obj.rayleigh_range = pi*obj.w0^2/obj.lambda;
        end
        
        function [density] = output_density(obj,x,y,z)
            if obj.normalized == 0
                normalization(obj);
                obj.normalized = 1;
            end
            
            [relrho,relz] = coord_convert(obj,x,y,z);
            
            w_sq = @(zrel) obj.w0.^2.*(1+zrel.^2./obj.rayleigh_range.^2);
            laserGauss = @(rhorel,zrel) obj.norm_factor./(pi.^2)./w_sq(zrel)./obj.time_res.^2./obj.c.^2.*exp(-2.*(rhorel.^2)./(w_sq(zrel))).*exp(-2.*(zrel-obj.c.*obj.cur_time).^2./obj.time_res.^2./obj.c.^2); % Norm Intensity
            
            density = laserGauss(relrho,relz);
        end
        
        function [relrho,relz] = coord_convert(obj,x,y,z)
            % initial adjustment to set the relative waist location at (0,0,0)
            xrel = x - obj.waist_loc(1);
            yrel = y - obj.waist_loc(2);
            zrel = z - obj.waist_loc(3);
            
            cur_rel = [xrel, yrel, zrel];
            
            % next step: dot products to align against or away from the Poynting vector
            
            relz = dot(cur_rel,obj.poynting_norm);
            relrho = norm(cur_rel - zrel*obj.poynting_norm);
        end
        
        function [norm_factor] = normalization(obj)
            % This function outputs in photon density (photons per volume)
            %[relrho,relz] = coord_convert(obj,x,y,z);
            
            %% NEED TO ESTABLISH A CUR_T POSITION FOR THE LASER BEAM
            
            w_sq = @(zrel) obj.w0.^2.*(1+zrel.^2./obj.rayleigh_range.^2);
            laserGauss = @(rhorel,zrel) 1./(pi.^2)./w_sq(zrel)./obj.time_res.^2./obj.c.^2.*exp(-2.*(rhorel.^2)./(w_sq(zrel))).*exp(-2.*(zrel-obj.c.*obj.cur_time).^2./obj.time_res.^2./obj.c.^2); % Norm Intensity
            total = integral2(@(x,y) 2.*pi.*x.*laserGauss(x,y), 0, Inf, -Inf, Inf);
            
            norm_factor = obj.E_p/(obj.h*obj.c/obj.lambda)/total;
            obj.norm_factor = norm_factor;
        end
    end
    
end
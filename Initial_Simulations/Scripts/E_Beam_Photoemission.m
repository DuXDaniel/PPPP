classdef E_Beam_Photoemission < handle   
    
    properties
        Ae; % Is variable to the position of the ebeam relative to the beam waist, will have to generalize the derivation from beam-waist position
        E_e; % total electron count, normalized to 1
        omeg_xze; % beam width of the ebeam at its smallest, is variable to the position in space (t)
        sig_te; % beam temporal width
        travel_dir; % vector of direction of travel
        travel_dir_norm;
        crossover_loc; % vector of waist location (x,y,z)
        crossover_angle; % gives the angle of the crossover
        time_res; % time stdev of the pulse
        cur_time;
        energy; % energy of the electrons
        vel; % velocity of the electron
    end
    
    properties (Constant)
        h = 6.626e-34; % J*s
    end
    
    methods
        function obj = E_Beam_Photoemission(pulse_energy,wavelength,dirTravel,polarization,waistPos,temporalRes,curTime)
            obj.E_e = 1;
            
            obj.travel_dir = dirTravel;
            obj.travel_dir_norm = obj.poynting./(sum(sqrt(obj.poynting.*obj.poynting)));
            obj.crossover_loc = waistPos;
            obj.time_res = temporalRes;
            obj.cur_time = curTime;
            
            obj.Ae = ;
            obj.energy = ;
            obj.omeg_xze = ; % based on curTime and slope of crossover
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
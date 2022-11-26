laser_res_range = logspace(3,5,3);
laser_width_range = logspace(3,6,10);
laser_pulse_range = logspace(1,3,10);

peak_phase_difference = zeros(10,10);

overall_min = pi;
overall_max = 0;

m=100;
cm_magma=magma(m);
cm_inferno=inferno(m);
cm_plasma=plasma(m);
cm_viridis=viridis(m);

for i = 1:length(laser_res_range)
    for l = 1:length(laser_pulse_range)
        for m = 1:length(laser_width_range)
            
            assign_var_name = ['Simul_Feynman_Func_V01','_',num2str(i),'_',num2str(l),'_',num2str(m)];
            try
                load(assign_var_name);
            catch
            end
            phase_dif = phase-min(min(phase));
            curmax = max(max(phase_dif));
            
            if curmax > overall_max
                overall_max = curmax;
            end
            
            if isnan(curmax)
                curmax = 0;
            end
            
            peak_phase_difference(l,m) = curmax;
            
            curmin = min(min(phase_dif));
            
            if curmin < overall_min
                overall_min = curmin;
            end
            
        end
    end
    %peak_phase_difference
    f = figure;
    %contourf(laser_width_range, laser_pulse_range, peak_phase_difference);
    if i == 1 || i == 2 
        peak_phase_difference(peak_phase_difference == max(max(peak_phase_difference))) = 0;
    end
    imagesc(peak_phase_difference);
    set(f,'Position',[0 0 1600 1600]);
    pbaspect([1 1 1]);
    xticks = [1 size(peak_phase_difference,1)];
    yticks = [1 size(peak_phase_difference,2)];
    xtick_labels = [laser_width_range(1) laser_width_range(end)];
    ytick_labels = [laser_pulse_range(1) laser_pulse_range(end)];
    set(gca,'TickLength',[0 0]);
    set(gca,'FontSize',16);
    set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
    set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
    %set(gca, 'YScale', 'log', 'FontSize', 26)
    %set(gca, 'XScale', 'log')
    xlabel('Beam Waist (nm)', 'FontSize',26);
    ylabel('Pulse Energy (nJ)', 'FontSize',26);
    title_name = [num2str(laser_res_range(i)/1000), ' ns'];
    colormap(cm_magma);
    cbh = colorbar;
    cbh.Ticks = [min(min(peak_phase_difference)) max(max(peak_phase_difference))];
    cbh.TickLabels = [round(min(min(peak_phase_difference))) max(max(peak_phase_difference))];
    cbh.FontSize = 16;
    title(title_name);
    print([title_name],'-dpng','-r600')
    close(f);
end

% for i = 1:length(laser_res_range)
%     for j = 1:length(e_res_range)
%         figure;
%         for l = 1:length(laser_pulse_range)
%             for m = 1:length(laser_width_range)
%
%                 assign_var_name = ['Simul_Feynman_Func_V01','_',num2str(i),'_',num2str(j),'_',num2str(l),'_',num2str(m)];
%                 title_name = ['set', num2str(l), num2str(m)];
%                 load(assign_var_name);
%                 subplot(10,10,(l-1)*(length(laser_pulse_range))+m);
%                 phase_dif = log(phase_dif);
%                 %phase_dif(1,1) = overall_max;
%                 %phase_dif(1,2) = overall_min;
%                 imagesc(phase_dif);
%                 colorbar;
%                 title(title_name);
%                 vpa([1 1 1]);
%             end
%         end
%     end
% end
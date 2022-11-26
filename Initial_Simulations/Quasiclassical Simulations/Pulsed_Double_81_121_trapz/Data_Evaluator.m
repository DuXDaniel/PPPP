laser_res_range = 10;
laser_width_range = logspace(3,6,2);
e_res_range = 10;
laser_pulse_range = logspace(-3,3,2);

peak_phase_difference = zeros(10,10);

overall_min = pi;
overall_max = 0;

m=100;
cm_magma=magma(m);
cm_inferno=inferno(m);
cm_plasma=plasma(m);
cm_viridis=viridis(m);

% for i = 1:length(laser_res_range)
%     for j = 1:length(e_res_range)
%         for l = 1:length(laser_pulse_range)
%             for m = 1:length(laser_width_range)
%                 
%                 assign_var_name = ['Simul_Feynman_Dual_Laser_Func_V01','_',num2str(i),'_',num2str(j),'_',num2str(l),'_',num2str(m)];
%                 try
%                 load(assign_var_name);
%                 catch
%                 end
%                 phase_dif = phase - min(min(phase));
%                 curmax = max(max(phase_dif));
%                 
%                 if curmax > overall_max
%                     overall_max = curmax;
%                 end
%                 
%                 if isnan(curmax)
%                     curmax = 0;
%                 end
%                 
%                 peak_phase_difference(l,m) = curmax;
%                 
%                 curmin = min(min(phase_dif));
%                 
%                 if curmin < overall_min
%                     overall_min = curmin;
%                 end
%                 
%             end
%         end
%         %peak_phase_difference
%         figure;
%         %contourf(laser_width_range, laser_pulse_range, peak_phase_difference);
%         imagesc(peak_phase_difference);
%         pbaspect([1 1 1]);
%         xticks = [1 size(peak_phase_difference,1)];
%         yticks = [1 size(peak_phase_difference,2)];
%         xtick_labels = [laser_width_range(1) laser_width_range(end)];
%         ytick_labels = [laser_pulse_range(1) laser_pulse_range(end)];
%         set(gca,'TickLength',[0 0]);
%         set(gca,'FontSize',16);
%         set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
%         set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
%         %set(gca, 'YScale', 'log', 'FontSize', 26)
%         %set(gca, 'XScale', 'log')
%         xlabel('Beam Waist (nm)', 'FontSize',26);
%         ylabel('Pulse Energy (nJ)', 'FontSize',26);
%         title_name = ['set', num2str(i), num2str(j)];
%         colormap(cm_magma);
%         cbh = colorbar;
%         cbh.Ticks = [min(min(peak_phase_difference)) max(max(peak_phase_difference))];
%         cbh.TickLabels = [round(min(min(peak_phase_difference))) max(max(peak_phase_difference))];
%         cbh.FontSize = 16;
%         title(title_name);
%     end
% end

% for i = 1:length(laser_res_range)
%     for j = 1:length(e_res_range)
%         set_name = ['Laser_Res_', num2str(laser_res_range(i)), 'ps_' 'E_Res_', num2str(e_res_range(j)), 'ps_'];
%         for l = 1:length(laser_pulse_range)
%             for m = 1:length(laser_width_range)
%                 
%                 assign_var_name = ['Quasiclassical_Func_Double_Laser_V01_SM_trapz_',num2str(i),'_',num2str(j),'_',num2str(l),'_',num2str(m)];
%                 title_name = ['Power_', num2str(l), 'Width_', num2str(laser_width_range(m)/1000), 'um'];
%                 load(assign_var_name);
%                 
%                 phase = Final_Phase_Data_Converter(laser_width_range(m), laser_res_range(i), e_res_range(j), laser_pulse_range(l), phase);
%                 if ~isnan(phase)
%                     
%                     f = figure;
%                     %subplot(3,2,(l-1)*(length(laser_width_range))+m);
%                     %                 phase_dif = log(phase_dif);
%                     %phase_dif(1,1) = overall_max;
%                     %phase_dif(1,2) = overall_min;
%                     phase_dif = phase - min(min(phase));
%                     imagesc(flipud(rot90(phase_dif,1)));
%                     xticks = [];
%                     yticks = [];
%                     xtick_labels = [];
%                     ytick_labels = [];
%                     set(gca,'TickLength',[0 0]);
%                     set(gca,'FontSize',16);
%                     set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
%                     set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
%                     set(gca, 'YDir','normal');
%                     colormap(cm_viridis);
%                     cbh = colorbar;
%                     cbh.Ticks = [min(min(phase_dif)) max(max(phase_dif))];
%                     cbh.TickLabels = [round(min(min(phase_dif))) max(max(phase_dif))];
%                     cbh.FontSize = 16;
%                     title([set_name title_name], 'Interpreter', 'none');
%                     set(gcf,'Position',[0 0 1600 1600]);
%                     pbaspect([1 1 1]);
%                     print([set_name title_name],'-dpng','-r600')
%                     close(f);
%                 end
%                 
%             end
%         end
%     end
% end

for i = 1:length(laser_res_range)
    for j = 1:length(e_res_range)
        set_name = ['Slices_','Laser_Res_', num2str(laser_res_range(i)), 'ps_' 'E_Res_', num2str(e_res_range(j)), 'ps_'];
        for l = 1:length(laser_pulse_range)
            for m = 1:length(laser_width_range)
                
                assign_var_name = ['Quasiclassical_Func_Double_Laser_V01_SM_trapz_',num2str(i),'_',num2str(j),'_',num2str(l),'_',num2str(m)];
                title_name = ['Power_', num2str(l), 'Width_', num2str(laser_width_range(m)), 'nm'];
                load(assign_var_name);
                
                if ~isnan(phase)
                    
                    f = figure;
                    %subplot(3,2,(l-1)*(length(laser_width_range))+m);
                    %                 phase_dif = log(phase_dif);
                    %phase_dif(1,1) = overall_max;
                    %phase_dif(1,2) = overall_min;
                    
                    for n = 1:size(phase,3)
                        subplot(11,11,n);
                        imagesc(flipud(rot90(phase(:,:,n),1)));
                    xticks = [];
                    yticks = [];
                    xtick_labels = [];
                    ytick_labels = [];
                    set(gca,'TickLength',[0 0]);
                    set(gca,'FontSize',16);
                    set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
                    set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
                    set(gca, 'YDir','normal');
                        colormap(cm_viridis);
                        %                         cbh = colorbar;
                        %                         cbh.Ticks = [min(min(phase_dif)) max(max(phase_dif))];
                        %                         cbh.TickLabels = [round(min(min(phase_dif))) max(max(phase_dif))];
                        %                         cbh.FontSize = 16;
                        %                         title([set_name title_name], 'Interpreter', 'none');
                        set(gcf,'Position',[0 0 1600 1600]);
                        pbaspect([1 1 1]);
                    end
                    print([set_name title_name],'-dpng','-r600')
                    close(f);
                end
            end
        end
    end
end
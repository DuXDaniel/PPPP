laser_res_range = logspace(1,3,2);
laser_width_range = logspace(3,6,2);
e_res_range = logspace(1,3,2);
laser_pulse_range = logspace(-3,3,3);

las_res_int = laser_res_range(1);
e_res_int = e_res_range(1);
las_width_int = laser_width_range(1);
las_pulse_int = laser_pulse_range(end);

voxel_gran_length = 3;
slice_gran_length = 3:2:103;
slice_string = [strsplit(num2str(slice_gran_length(1:35).^2),' '), '10000'];
% f1 = figure;
f2 = figure;
% f3 = figure;

m=100;
cm_magma=magma(m);
cm_inferno=inferno(m);
cm_plasma=plasma(m);
cm_viridis=viridis(m);

sizeofgrid = ceil(sqrt(9));

for i = 1:length(voxel_gran_length)
    for j = 1:9%length(slice_gran_length)
        
%         figure(f1);
%         
%         assign_var_name = ['Quasiclassical_Func_V01_SM_VarVox_1_1_1_3_2_',num2str(j)];
%         try
%             load(assign_var_name);
%             phase_shifts = phase - min(min(phase));
%             subplot(sizeofgrid,sizeofgrid,(i-1)*length(slice_gran_length) + j);
%         catch
%         end
%         
%         imagesc(flipud(rot90(phase_shifts,-1)));
%         title(slice_string(j));
%         xticks = [];
%         yticks = [];
%         xtick_labels = [];
%         ytick_labels = [];
%         set(gca,'TickLength',[0 0]);
%         set(gca,'FontSize',16);
%         set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
%         set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
%         colormap(cm_viridis);
%         %         cbh = colorbar;
%         %         cbh.Ticks = [min(min(phase)) max(max(phase))];
%         %         cbh.TickLabels = [round(min(min(phase))) max(max(phase))];
%         %         cbh.FontSize = 16;
%         set(gcf,'Position',[0 0 1600 1600]);
%         pbaspect([1 1 1]);
        
        figure(f2);
        
        assign_var_name = ['Quasiclassical_Func_Double_Laser_V01_SM_VarVox_1_1_2_3_2_',num2str(j), '_1'];
        try
            load(assign_var_name);
            phase_shifts = phase - min(min(phase));
            subplot(sizeofgrid,sizeofgrid,(i-1)*length(slice_gran_length) + j);
        catch
            break;
        end
        
        imagesc(flipud(rot90(phase_shifts,1)));
        title(slice_string(j));
        xticks = [];
        yticks = [];
        xtick_labels = [];
        ytick_labels = [];
        set(gca,'TickLength',[0 0]);
        set(gca,'FontSize',16);
        set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
        set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
        set(gca, 'YDir', 'normal');
        colormap(cm_viridis);
        %         cbh = colorbar;
        %         cbh.Ticks = [min(min(phase)) max(max(phase))];
        %         cbh.TickLabels = [round(min(min(phase))) max(max(phase))];
        %         cbh.FontSize = 16;
        set(gcf,'Position',[0 0 1600 1600]);
        pbaspect([1 1 1]);
        
%         figure(f3);
%         
%         j
%         
%         assign_var_name = ['Feynman_Func_Double_Laser_V01_SM_VarVox_1_1_1_3_2_',num2str(j), '_1'];
%         try
%             load(assign_var_name);
%             phase_shifts = phase - min(min(phase));
%             subplot(sizeofgrid,sizeofgrid,(i-1)*length(slice_gran_length) + j);
%         catch
%             break;
%         end
%         
%         imagesc(flipud(rot90(phase_shifts,1)));
%         title(max(max(phase_shifts)));
%         xticks = [];
%         yticks = [];
%         xtick_labels = [];
%         ytick_labels = [];
%         set(gca,'TickLength',[0 0]);
%         set(gca,'FontSize',16);
%         set(gca, 'XTick', xticks, 'XTickLabel', xtick_labels);
%         set(gca, 'YTick', yticks, 'YTickLabel', ytick_labels);
%         set(gca, 'YDir', 'normal');
%         colormap(cm_viridis);
%         %         cbh = colorbar;
%         %         cbh.Ticks = [min(min(phase)) max(max(phase))];
%         %         cbh.TickLabels = [round(min(min(phase))) max(max(phase))];
%         %         cbh.FontSize = 16;
%         set(gcf,'Position',[0 0 1600 1600]);
%         pbaspect([1 1 1]);
       
        pause(0.01);
    end
end

% figure(f1);
% print('Quasi_Single_Laser_1_1_1_3_Slice_Granularity','-dpng','-r600')
figure(f2);
print('Quasi_Double_Laser_1_1_2_3_Slice_Granularity','-dpng','-r600')
% figure(f3);
% print('Feynman_Double_Laser_1_1_1_3_Slice_Granularity','-dpng','-r600')
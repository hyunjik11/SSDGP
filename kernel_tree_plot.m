function kernel_tree_plot(kernel_dict,m_values,y_lim,directory)
keys = kernel_dict.keys; 
for key_ind = 1:length(keys)
    key=keys{key_ind};
    kernel_cell = kernel_dict(key);
    lb = kernel_cell{1}; ub = kernel_cell{4}; ne = kernel_cell{5};
    figure();
    set(gca,'fontsize',18)
    hold on
    x_idx = 1:length(m_values);
    xlim([0.5,6.5])
    ylim(y_lim);
    %plot(x_idx,ne*ones(size(m_values)));
    midpt = 0.5*(ub + lb);
    myzero = zeros(size(ub));
    errorbar(x_idx,midpt,midpt-lb,midpt-ub,'.','MarkerSize',0.1,'LineWidth',4)
    %plot(x_idx,ub,'LineWidth',3);
    %plot(x_idx,lb,'LineWidth',1);
    set(gca,'XTick',[1 2 3 4 5 6]);
    set(gca,'XTickLabel',[10 20 40 80 160 320]);
    xlabel('m')
    %ylabel('negative energy')
    title(key)
    %legend('UB','fullGP','LB')
    grid on
    grid minor
    hold off
    file_name=strcat(directory,key,'.png');
    file_name = regexprep(file_name,'+','_plus_');
    file_name = regexprep(file_name,'*','_times_');
    saveas(gcf,file_name)
end

end
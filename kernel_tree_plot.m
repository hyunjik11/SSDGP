function kernel_tree_plot(kernel_dict,m_values)
keys = kernel_dict.keys; 
for key_ind = 1:length(keys)
    key=keys{key_ind};
    kernel_cell = kernel_dict(key);
    lb = kernel_cell{1}; ub = kernel_cell{4}; ne = kernel_cell{7};
    figure();
    hold on
    x_idx = 1:length(m_values);
    plot(x_idx,ub);
    plot(x_idx,ne*ones(size(m_values)));
    plot(x_idx,lb);
    set(gca,'XTick',[1 2 3 4 5 6]);
    set(gca,'XTickLabel',[10 20 40 80 160 320]);
    xlabel('m')
    ylabel('negative energy')
    title(key)
    legend('UB','fullGP','LB')
    hold off
end

end
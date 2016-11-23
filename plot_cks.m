function plot_cks(kernel_buffer_history,kernel_top)
    nk = length(kernel_buffer_history); % number of kernels in buffer_history
    Y=nk:-1:1;
    X = zeros(nk,1);
    keys = cell(nk,1);
    for kernel_ind = 1:nk
        kernel = kernel_buffer_history(kernel_ind);
        X(kernel_ind) = kernel.bic;
        keys{nk-kernel_ind+1} = kernel.key;
    end
    figure();
    scatter(X,Y,'bx');
    xlabel('BIC')
    ylim([0.5,nk+0.5])
    set(gca,'YTick',1:nk);
    set(gca,'YTickLabel',keys);
    title_str = ['top = ' kernel_top.key];
    title(title_str);
    grid on
    grid minor
end
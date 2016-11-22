function plot_skd(kernel_buffer_history,m,S)
    nk = length(kernel_buffer_history); % number of kernels in buffer_history
    Y=nk:-1:1;
    Y=[Y;Y];
    X = zeros(2,nk);
    keys = cell(nk,1);
    for kernel_ind = 1:nk
        kernel = kernel_buffer_history(kernel_ind);
        X(1,kernel_ind) = kernel.lb;
        X(2,kernel_ind) = kernel.ub;
        keys{nk-kernel_ind+1} = kernel.key;
    end
    plot(X,Y,'LineWidth',3);
    xlabel('BIC')
    ylim([1-nk/2,nk+nk/2])
    hold on
    scatter(X(:),Y(:),20,'bx')
    hold off
    set(gca,'YTick',1:nk);
    set(gca,'YTickLabel',keys);
    title_str = ['m = ' num2str(m) ', S = ' num2str(S)];
    title(title_str);
    grid on
    grid minor
end
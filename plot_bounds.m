function plot_bounds(kernel_buffer_history,kernel_top,m,S,x,y)
    nk = length(kernel_buffer_history); % number of kernels in buffer_history
    Y=nk:-1:1;
    Y=[Y;Y];
    X = zeros(2,nk);
    keys = cell(nk,1);
    exact = zeros(1,nk);
    for kernel_ind = 1:nk
        kernel = kernel_buffer_history(kernel_ind);
        X(1,kernel_ind) = kernel.lb;
        X(2,kernel_ind) = kernel.ub;
        gpcf = kernel.gp_var.cf;
        lik = kernel.gp_var.lik;
        tic
        [bic, ~] = gpeval(x,y,gpcf,lik);
        toc
        exact(kernel_ind) = bic;
        keys{nk-kernel_ind+1} = kernel.key;
        if strcmp(kernel.key,kernel_top.key)
            keys{nk-kernel_ind+1} = ['-> ' kernel.key];
        end
    end
    plot(X,Y,'LineWidth',3);
    xlabel('BIC')
    ylim([0.5,nk+0.5])
    hold on
    scatter(exact,nk:-1:1,20,'bx')
    hold off
    set(gca,'YTick',1:nk);
    set(gca,'YTickLabel',keys);
    title_str = ['m = ' num2str(m) ', S = ' num2str(S) ', top = ' kernel_top.key];
    title(title_str);
    grid on
    grid minor
    
end
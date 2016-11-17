function kernel_tree_plot_lb(x,y,kernel_dict,m_values,y_lim)%,directory)
% function to compare varLB optimised using rand init for m=max(m_values)
% against varLB building up from small m.
num_iter = 10;
POOL=parpool('local',num_iter);
keys = kernel_dict.keys; 
for key_ind = 1:length(keys)
    key=keys{key_ind};
    kernel_cell = kernel_dict(key);
    lb = kernel_cell{1}; ne = kernel_cell{5};
    gp = kernel_cell{6};
    lb_table = zeros(num_iter,1);
    parfor seed = 1:num_iter
        m=max(m_values);
        rng(seed);
        xu = datasample(x,m,1,'Replace',false);
        [gpcf, lik] = reinitialise_kernel(gp.cf{1},x,y);
        lb_table(seed) = lbfunction(x,y,xu,gpcf,lik);
    end
    figure();
    set(gca,'fontsize',18)
    hold on
    x_idx = 1:length(m_values);
    xlim([0.5,6.5])
    ylim(y_lim);
    plot(x_idx,ne*ones(size(m_values)),'r','LineWidth',3);
    plot(x_idx,lb,'y','LineWidth',2);
    plot(x_idx,max(lb_table)*ones(size(m_values)),'g','LineWidth',1);
    set(gca,'XTick',[1 2 3 4 5 6]);
    set(gca,'XTickLabel',[10 20 40 80 160 320]);
    xlabel('m')
    %ylabel('negative energy')
    title(key)
    %legend('UB','fullGP','LB')
    hold off
%     file_name=strcat(directory,key,'.png');
%     file_name = regexprep(file_name,'+','_plus_');
%     file_name = regexprep(file_name,'*','_times_');
%     saveas(gcf,file_name)
    fprintf([key ' done \n']);
end
delete(POOL);
end
        
function [lb, gp_var] = lbfunction(x,y,xu,gpcf,lik)
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', xu);
    gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp_var = gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
    if gp_var.lik.sigma2 > 1e-8
        lb = gp_e([],gp_var,x,y);
        lb = -lb;
    else lb = nan;
    end
end
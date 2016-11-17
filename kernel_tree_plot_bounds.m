function kernel_tree_plot_bounds(x,y,kernel_dict,m_values,y_lim) %directory
% function to see what happens to LB&UB when hyp fixed to optimal ones
% found by full GP and m varies. Use error bars for different ind pt inits
num_iter = 10;
%POOL=parpool('local',num_iter);
keys = kernel_dict.keys; 
ind=0;
for key_ind = 1:length(keys)
    ind=ind+1;
    key=keys{key_ind};
    kernel_cell = kernel_dict(key);
    ne = kernel_cell{5}; gp = kernel_cell{6};
    lb_table=zeros(num_iter,length(m_values));
    ub_table=zeros(num_iter,length(m_values));
    for i=1:length(m_values)
        m = m_values(i);
        parfor seed = 1:num_iter
            rng(seed);
            xu = datasample(x,m,1,'Replace',false);
            gp_var = gp_set(gp,'type', 'VAR','X_u', xu);
            lb = gp_e([],gp_var,x,y);
            lb = -lb;
            lb_table(seed,i) = lb;
            ub_table(seed,i) = ubfunction(x,y,gp_var,'PIC');
        end
    end
    subplot(3,7,ind)
    set(gca,'fontsize',10)
    hold on
    x_idx = 1:length(m_values);
    xlim([0.5,6.5])
    ylim(y_lim);
    errorbar(x_idx,mean(ub_table),std(ub_table));
    plot(x_idx,ne*ones(size(m_values)),'LineWidth',2);
    errorbar(x_idx,mean(lb_table),std(lb_table));
    set(gca,'XTick',[1 2 3 4 5 6]);
    set(gca,'XTickLabel',[10 20 40 80 160 320]);
    xlabel('m')
    %ylabel('negative energy')
    title(key)
    %legend('UB','fullGP','LB')
    hold off
    %file_name=strcat(directory,key,'.png');
    %file_name = regexprep(file_name,'+','_plus_');
    %file_name = regexprep(file_name,'*','_times_');
    %saveas(gcf,file_name)
end
%delete(POOL);
end


function ub = ubfunction(x,y,gp_var,precond)
    warning('off','all');
    xu = gp_var.X_u;
    m = size(xu,1);
    n = size(x,1);
    signal_var = gp_var.lik.sigma2;
    K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
    L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
    L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
    K_naive=L'*L;
    A=L*L'+signal_var*eye(m);
    L_naive=chol(A);
    naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
    [K,C]=gp_trcov(gp_var,x);
    switch precond
        case 'None'
            [~,~,~,~,~,cg_obj]=cgs_obj(C,y,[],m);
            nip_ub = min(cg_obj);
        case 'Nystrom'
            %function handle which gives (K_naive+signal_var*eye(n))\x
            myfun = @(w) (w-L'*(A\(L*w)))/signal_var;
            [~,~,~,~,~,pcg_obj]=cgs_obj(C,y,[],m,myfun);
            nip_ub = min(pcg_obj);
        case 'FIC'
            dinv=1./(diag(K)-diag(K_naive)+signal_var);
            Dinv=diag(dinv); %D=diag(K-K_naive)+signal_var*eye(n)
            Af=L*Dinv*L'+eye(m);
            %function handle which gives (K_fic+signal_var*eye(n))\x
            myfunf = @(w) (w-L'*(Af\(L*(w.*dinv)))).*dinv;
            [~,~,~,~,~,pcg_objf]=cgs_obj(C,y,[],m,myfunf);
            nip_ub = min(pcg_objf);
        case 'PIC'
            %funcion handle which gives (K_pic+signal_var*eye(n))\x
            [~,invfun]=blockdiag(K-K_naive,m,signal_var); %invfun is fhandle which gives (Kb+signal_var*eye(n))\w
            Ap=L*invfun(L')+eye(m);
            myfunp = @(w) invfun(w-L'*(Ap\(L*invfun(w))));
            [~,~,~,~,~,pcg_objp]=cgs_obj(C,y,[],m,myfunp);
            nip_ub = min(pcg_objp);
        otherwise
            error('precond is invalid')
    end
    logprior = 0;
    ncf = length(gp_var.cf); % number of hyp
    for i=1:ncf
        gpcf = gp_var.cf{i};
        logprior = logprior + gpcf.fh.lp(gpcf);
    end
    lik = gp_var.lik;
    logprior = logprior + lik.fh.lp(lik);
    ub = naive_nld + nip_ub - 0.5*n*log(2*pi) + logprior;
end
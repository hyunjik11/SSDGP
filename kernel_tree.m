function [kernel_dict, kernel_dict_debug] = kernel_tree(x,y,final_depth,num_iter,m_values,seed,fullgp,precond)
% function to get LB, UB and fullGP ml for each kernel in the space of
% compositional kernels up to depth=final_depth 
% with numiter rand init for each kernel.
% gpcf_dict contains the kernels with optimal hyp for each of LB, UB, fullGP
% base_kernels is dictionary with string keys and gpcf values
% base_kernels should contain 3*ndims kernels where ndims is dim of x values
% fullgp is a boolean for whether fullgp optim should be done or not
% precond is the preconditioner used for PCG for ub - can take values from:
% 'None','Nystrom','FIC','PIC'
base_kernels = construct_base_kernels(x,y);

%%% initialise dictionary with key = kernel type, 
%%% value = cell of lb(+gpcf,lik),ub,fullgp(+gpcf,lik) (size=7, order = lb, ub, fullGP)
kernel_dict = containers.Map('KeyType','char','ValueType','any');

%%% intialise dictionary of kernels at current depth with key = kernel type,
%%% value = cell of lb(+gpcf,lik,indices of indpts),ub,fullgp(+gpcf,lik) (size=8, order = lb, ub, fullGP)
kernel_dict_depth = containers.Map('KeyType','char','ValueType','any');

% debug contains lb(+gpcf,lik,indices_indpts), ub(+gpcf,lik), fullgp(+gpcf,lik) for all iterations
kernel_dict_debug = containers.Map('KeyType','char','ValueType','any');

rng(seed);
[n,~] = size(x); % n is n_data, ndims is dimensionality of inputs
%%% TODO: fix algorithm for ndims > 1 (multidim inputs e.g. concrete)

nm = length(m_values);

%%% kernel search
for depth = 1:final_depth
    kernel_dict_depth_new = containers.Map('KeyType','char','ValueType','any');
    if depth == 1 %%% for first depth, branch factor = 3 * ndims
        for key_ind = 1:length(base_kernels.keys)
            keys = base_kernels.keys; key = keys{key_ind}; % name of kernel
            val = base_kernels(key); % gpcf in base kernel
            lb_table = zeros(num_iter,nm); % stores lb for all iter
            ub_table = zeros(1,nm); % stores ub for all m
            lb_gpcf_cell = cell(num_iter,nm); % stores lb gpcf for all iter
            lb_lik_cell = cell(num_iter,nm); % stores lb lik for all iter
            if fullgp
                gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                gp_gpcf_cell = cell(num_iter,1); % stores fullgp gpcf for all iter
                gp_lik_cell = cell(num_iter,1); % stores fullgp lik for all iter
            end
            idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
            idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
            [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
            for i = 1:nm
                m = m_values(i);
                parfor iter = 1:num_iter % change to parfor for parallel
                    %rng(iter);                    
                    % optim for lb
                    if i==1 || iter <= num_iter/2 % use rand init of hyp for all iter of first m & half iters for other m's
                        [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
                        [gpcf, lik] = reinitialise_kernel(val,x,y);
                        [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf,lik);
                    else % keep optimal ind pts from previous m, and also keep the hyp
                        weights = 1e-10*ones(1,n); %weights for sampling
                        weights(idx_u)=1; %make sure samples idx_u are included
                        [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                        [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf_best,lik_best);
                    end
                    
                    % optim for fullgp
                    if fullgp && (i==1)
                        [gp_table(iter),gp_gpcf_cell{iter},gp_lik_cell{iter}] = gpfunction(x,y,gpcf,lik);
                    end
                end
                [~,ind] = max(lb_table(:,i));
                idx_u = idx_cell{ind,i}; %indices of subset for best LB
                % find ub for hyp from best LB
                xu = x(idx_u,:);
                gpcf_best = lb_gpcf_cell{ind,i};
                lik_best = lb_lik_cell{ind,i};
                gp_var = gp_set('type', 'VAR', 'lik', lik_best, 'cf',gpcf_best,'X_u', xu);
                gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
                ub_table(i) = ubfunction(x,y,gp_var,precond);
            end
            % gather best results
            [lb,ind] = max(lb_table,[],1);
            lb_gpcf = cell(1,nm); lb_lik = cell(1,nm); lb_idx=cell(1,nm);
            for i=1:nm
                lb_gpcf{i} = lb_gpcf_cell{ind(i),i};
                lb_lik{i} = lb_lik_cell{ind(i),i};
                lb_idx{i} = idx_cell{ind(i),i};
            end
            
            if fullgp
                [ml,ind] = max(gp_table);
                gp_gpcf = gp_gpcf_cell{ind};
                gp_lik = gp_lik_cell{ind};
            end
            
            % store all data in debug_cell, value for kernel_dict_debug
            debug_cell = {};
            debug_cell{1} = lb_table; debug_cell{2} = lb_gpcf_cell; debug_cell{3} = lb_lik_cell;
            debug_cell{4} = idx_cell; debug_cell{5} = ub_table; 
         
            % store best results in gpcf_dict_depth
            depth_cell = {};
            depth_cell{1} = lb; depth_cell{2} = lb_gpcf; depth_cell{3} = lb_lik;
            depth_cell{4} = lb_idx; depth_cell{5} = ub_table; 
            if fullgp
                debug_cell{6} = gp_table; debug_cell{7} = gp_gpcf_cell; debug_cell{8} = gp_lik_cell;
                depth_cell{6} = ml; depth_cell{7} = gp_gpcf; depth_cell{8} = gp_lik;
            end
            kernel_dict_debug(key) = debug_cell;
            kernel_dict_depth_new(key) = depth_cell;
            fprintf([key ' done \n']);
        end
        
    else %%% for depth > 1, branch factor = 6*ndims
        for key_ind = 1:length(kernel_dict_depth.keys)
            keys = kernel_dict_depth.keys; key = keys{key_ind};
            depth_cell = kernel_dict_depth(key);
            val = depth_cell{2}{nm}; % gpcf with hyp giving best lb for parent kernel with max m
            lik = depth_cell{3}{nm}; % lik with hyp giving best lb for parent kernel with max m
            for base_key_ind = 1:length(base_kernels.keys)
                base_keys = base_kernels.keys; key_base = base_keys{base_key_ind};
                val_base = base_kernels(key_base);
                for comp = 0:1
                    if comp ==0 % kernel in previous depth + base kernel
                        key_new = ['(' key ')+' key_base];
                    else % kernel in previous depth * base kernel
                        key_new = ['(' key ')*' key_base];
                    end
                    
                    % now that we've selected kernel, 
                    % carry out compositional search as before
                    lb_table = zeros(num_iter,nm); % stores lb for all iter
                    ub_table = zeros(1,nm); % stores ub for all iter
                    lb_gpcf_cell = cell(num_iter,nm); % stores lb gpcf for all iter
                    lb_lik_cell = cell(num_iter,nm); % stores lb lik for all iter
                    if fullgp
                        gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                        gp_gpcf_cell = cell(num_iter,1); % stores fullgp gpcf for all iter
                        gp_lik_cell = cell(num_iter,1); % stores fullgp lik for all iter
                    end
                    idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
                    idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
                    [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
                    for i = 1:nm
                        m = m_values(i);
                        parfor iter = 1:num_iter
                            %rng(iter);
                            % optim for lb
                            if i==1 || iter<=num_iter/2  % get optimal hyp from previous depth kernels, with new ind pts and hyps for current depth kernel
                                [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
                                [val_base_new,~] = reinitialise_kernel(val_base,x,y);
                                if comp==0 % kernel in previous depth + base kernel
                                    gpcf_new = gpcf_sum('cf',{val,val_base_new});
                                else % kernel in previous depth * base kernel
                                    gpcf_new = gpcf_prod('cf',{val,val_base_new});
                                end
                                [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf_new,lik);

                            else % for half the iter, keep optimal ind pts and hyp from previous m
                                weights = 1e-10*ones(1,n); %weights for sampling
                                weights(idx_u)=1; %make sure samples idx_u are included
                                [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                                [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf_best,lik_best);
                            end
                            
                            % optim for fullgp
                            if fullgp && (i==1)
                                [gp_table(iter),gp_gpcf_cell{iter},gp_lik_cell{iter}] = gpfunction(x,y,gpcf_new,lik);
                            end
                        end
                        [~,ind] = max(lb_table(:,i));
                        idx_u = idx_cell{ind,i}; %indices of subset for best LB
                        % find ub for hyp from best LB
                        xu = x(idx_u,:);
                        gpcf_best = lb_gpcf_cell{ind,i};
                        lik_best = lb_lik_cell{ind,i};
                        gp_var = gp_set('type', 'VAR', 'lik', lik_best, 'cf',gpcf_best,'X_u', xu);
                        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
                        ub_table(i) = ubfunction(x,y,gp_var,precond);
                    end
                    % gather best results
                    [lb,ind] = max(lb_table,[],1);
                    lb_gpcf = cell(1,nm); lb_lik = cell(1,nm); lb_idx=cell(1,nm);
                    for i=1:nm
                        lb_gpcf{i} = lb_gpcf_cell{ind(i),i};
                        lb_lik{i} = lb_lik_cell{ind(i),i};
                        lb_idx{i} = idx_cell{ind(i),i};
                    end
                    
                    if fullgp
                        [ml,ind] = max(gp_table);
                        gp_gpcf = gp_gpcf_cell{ind};
                        gp_lik = gp_lik_cell{ind};
                    end
                    
                    % store all data in debug_cell, value for kernel_dict_debug
                    debug_cell = {};
                    debug_cell{1} = lb_table; debug_cell{2} = lb_gpcf_cell; debug_cell{3} = lb_lik_cell;
                    debug_cell{4} = idx_cell; debug_cell{5} = ub_table; 
                    
                    % store best results in gpcf_dict_depth
                    depth_cell = {};
                    depth_cell{1} = lb; depth_cell{2} = lb_gpcf; depth_cell{3} = lb_lik;
                    depth_cell{4} = lb_idx; depth_cell{5} = ub_table; 
                    if fullgp
                        debug_cell{6} = gp_table; debug_cell{7} = gp_gpcf_cell; debug_cell{8} = gp_lik_cell;
                        depth_cell{6} = ml; depth_cell{7} = gp_gpcf; depth_cell{8} = gp_lik;
                    end
                    kernel_dict_debug(key_new) = debug_cell;
                    kernel_dict_depth_new(key_new) = depth_cell;
                    fprintf([key_new ' done \n']);
                end
            end
        end
    end
    kernel_dict = [kernel_dict; kernel_dict_depth_new];
    kernel_dict_depth = kernel_dict_depth_new;
    fprintf('depth %d done\n',depth);
end

end

function base_kernels = construct_base_kernels(x,y)
    ndims = size(x,2);
    base_set = {'SE','LIN','PER'};
    l = length(base_set);
    if ndims > 1
        keySet = cell(1,ndims*l);
        valueSet = cell(1,ndims*l);
        counter = 1;
        for num_base = 1:l
            base_ker = base_set{num_base};
            for dim = 1:ndims
                key = strcat(base_ker,num2str(dim));
                keySet{counter} = key;
                switch num_base % needs to be modified if base_set modified
                    case 1
                        valueSet{counter} = se_init(x,y,dim);
                    case 2
                        valueSet{counter} = lin_init(dim);
                    case 3
                        valueSet{counter} = per_init(x,y,dim);
                    otherwise
                        error('base_set larger than 3')
                end
                counter = counter +1;
            end
        end
    else
        keySet = base_set;
        valueSet = {se_init(x,y),lin_init(),per_init(x,y)};
    end
    base_kernels=containers.Map(keySet,valueSet);
end

function [gpcf, lik] = reinitialise_kernel(gpcf, x, y)
    % function to reinitialise gpcf of base kernels and lik
    if size(x,2) > 1
        dim = gpcf.selectedVariables;
        switch gpcf.type
            case 'gpcf_sexp'
                gpcf = se_init(x,y,dim);
            case 'gpcf_linear'
                gpcf = lin_init(dim);
            case 'gpcf_periodic'
                gpcf = per_init(x,y,dim);
            otherwise
                error('gpcf of invalid type - not in base_kernels');
        end
    else
        switch gpcf.type
            case 'gpcf_sexp'
                gpcf = se_init(x,y);
            case 'gpcf_linear'
                gpcf = lin_init();
            case 'gpcf_periodic'
                gpcf = per_init(x,y);
            otherwise
                error('gpcf of invalid type - not in base_kernels');
        end
    end
    lik = lik_init(y);
end

function [lb, gpcf, lik] = lbfunction(x,y,xu,gpcf,lik)
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', xu);
    gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp_var = gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
    lb = gp_e([],gp_var,x,y);
    lb = -lb;
    gpcf = gp_var.cf{1};
    lik = gp_var.lik;
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

function [ne, gpcf, lik] = gpfunction(x,y,gpcf,lik)
    gp=gp_set('lik',lik,'cf',gpcf);
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp = gp_optim(gp,x,y,'opt',opt,'optimf',@fminscg);
    [ne, ~ , ~] = gp_e([],gp,x,y);
    ne = -ne;
    gpcf = gp.cf{1};
    lik = gp.lik;
end
function [kernel_buffer,kernel_buffer_history, kernel_top, kernel_top_history] = skd(x,y,final_depth,num_iter,m_values,seed,S,precond)
% function to carry out scalable kernel discovery on inputs x, outputs y
% up to depth = final_depth
% with num_iter rand inits for each kernel
% kernel_buffer is a struct array of size S which contains the top S kernels.
% kernel_buffer_history is a struct array that contains the kernels newly added to kernel_buffer at each depth.
% kernel_top is the struct for the kernel found by skd.
% kernel_top_history is a struct array of kernels that contains the kernel
% selected at each depth.
% Each kernel is a struct with fields: key, lb, ub (for BIC), gp_var, indices (of
% ind pts)
% precond is the preconditioner used for PCG for ub - can take values from:
% 'None','Nystrom','FIC','PIC'
base_kernels = construct_base_kernels(x,y); % create a dictionary of base kernels

%%% initialise kernel_buffer
kernel_buffer = struct('key',{},'lb',{},'ub',{},'gp_var',{},'indices',{});

%%% initialise kernel_buffer_history
kernel_buffer_history = struct('key',{},'lb',{},'ub',{},'gp_var',{},'indices',{});

%%% initialise kernel_top
kernel_top = struct('key',[],'lb',[],'ub',[],'gp_var',[],'indices',[]);

%%% initialise kernel_top_history
kernel_top_history = struct('key',{},'lb',{},'ub',{},'gp_var',{},'indices',{});

%%% initialise kernel_new, the cell array of kernels newly added to
%%% kernel_buffer at each depth - for the following depth, the three will
%%% only grow on these kernels
kernel_new = struct('key',{},'lb',{},'ub',{},'gp_var',{},'indices',{});

rng(seed);
[n,~] = size(x); % n is n_data
nm = length(m_values);

for depth = 1:final_depth
    if depth == 1
        for key_ind = 1:length(base_kernels.keys)
            keys = base_kernels.keys; key = keys{key_ind}; % name of kernel
            val = base_kernels(key); % gpcf in base kernel
            lb_table = zeros(num_iter,nm); % stores lb for all iter
            ub_table = zeros(1,nm); % stores ub for all m
            gp_var_cell = cell(num_iter,nm); % stores lb gp_var for all iter
            idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
            idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
            [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
            for i = 1:nm
                m = m_values(i);
                parfor iter = 1:num_iter % change to parfor for parallel
                    rng(iter);                    
                    %%% optim for lb
                    if i==1 || iter <= 0.8*num_iter % use rand init of hyp for all iter of first m & 4/5 of iters for other m's
                        [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
                        [gpcf, lik] = reinitialise_kernel(val,x,y);
                        [lb_table(iter,i),gp_var_cell{iter,i}] = lbfunction(x,y,xu,gpcf,lik);
                    else % for 1/5 of iter, keep optimal ind pts from previous m, and also keep the hyp
                        weights = 1e-10*ones(1,n); %weights for sampling
                        weights(idx_u)=1; %make sure samples idx_u are included
                        [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                        [lb_table(iter,i),gp_var_cell{iter,i}] = lbfunction(x,y,xu,gpcf_best,lik_best);
                    end
                end
                [~,ind] = max(lb_table(:,i));
                idx_u = idx_cell{ind,i}; %indices of subset for best LB
                
                %%% find ub for hyp from best LB
                gp_var_best = gp_var_cell{ind,i};
                gpcf_best = gp_var_best.cf{1};
                lik_best = gp_var_best.lik;
                ub_table(i) = ubfunction(x,y,gp_var_best,precond);
            end
            %%% gather best result for m=max(m_values), and store in kernel
            [lb,ind] = max(lb_table(:,nm));
            ub = ub_table(nm);
            gp_var = gp_var_cell{ind,nm};
            indices = idx_cell{ind,nm};
            kernel = struct('key',key,'lb',lb,'ub',ub,'gp_var',gp_var,'indices',indices);
            
            %%% compare kernel with previous kernels
            n_buffer = length(kernel_buffer);
            if n_buffer == 0 % buffer is empty
                kernel_buffer(1) = kernel; kernel_top = kernel;
            elseif ub < kernel_top.lb % kernel interval strictly below top kernel interval
                % ignore kernel
            elseif lb < kernel_top.lb % kernel interval overlaps with top kernel interval, but has lower lb than top_kernel
                [buffer_min_val, buffer_min_ind] = findmin(kernel_buffer);
                if n_buffer < S % buffer not full
                    kernel_buffer(n_buffer+1) = kernel;
                elseif lb > buffer_min_val % if kernel has higher lb than some kernel in buffer
                    kernel_buffer(buffer_min_ind) = kernel;
                end
            else % kernel.lb > kernel_top.lb
                kernel_top = kernel;
                %%% compare kernels in buffer to new kernel_top, and see if
                %%% they should remain or be deleted
                for buffer_ind = 1:length(kernel_buffer)
                    buffer_kernel = kernel_buffer(buffer_ind);
                    if buffer_kernel.ub < lb % if kernel in buffer has strictly lower interval than kernel_top
                        kernel_buffer(buffer_ind) = [];
                    end
                end
                n_buffer_new = length(kernel_buffer);
                if n_buffer_new < S % if buffer is not full
                    kernel_buffer(n_buffer_new+1) = kernel;
                else % buffer full, so replace the buffer kernel with the lowest lb
                    [~, buffer_min_ind] = findmin(kernel_buffer);
                    kernel_buffer(buffer_min_ind) = kernel;
                end
            end
            fprintf([key ' done. lb=%4.2f, ub = %4.2f \n'],lb,ub);
        end
        kernel_new = kernel_buffer;
    else % if depth > 1
        if isempty(kernel_new) % no new kernels found in search
            return
        else
        kernel_buffer_old = kernel_buffer; % need for comparing with kernel_buffer after search at current depth
        for parent_ind = 1:length(kernel_new)
            key = kernel_new(parent_ind).key;
            val = kernel_new(parent_ind).gp_var.cf{1};
            lik = kernel_new(parent_ind).gp_var.lik;
            for base_key_ind = 1:length(base_kernels.keys)
                base_keys = base_kernels.keys; key_base = base_keys{base_key_ind};
                val_base = base_kernels(key_base);
                for comp = 0:1 % select kernel
                    if comp ==0 % kernel in previous depth + base kernel
                        key_new = ['(' key ')+' key_base];
                    else % kernel in previous depth * base kernel
                        key_new = ['(' key ')*' key_base];
                    end
                    lb_table = zeros(num_iter,nm); % stores lb for all iter
                    ub_table = zeros(1,nm); % stores ub for all iter
                    gp_var_cell = cell(num_iter,nm); % stores lb gp_var for all iter
                    idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
                    idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
                    [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
                    for i = 1:nm
                        m = m_values(i);
                        parfor iter = 1:num_iter
                            rng(iter);
                            %%% optim for lb
                            if i==1 || iter<=0.8*num_iter  
                                % for m_min, or for 4/5 of the iter, split into half and half:
                                [val_base_new,~] = reinitialise_kernel(val_base,x,y);
                                if comp==0 % kernel in previous depth + base kernel
                                    gpcf_new = gpcf_sum('cf',{val,val_base_new});
                                else % kernel in previous depth * base kernel
                                    gpcf_new = gpcf_prod('cf',{val,val_base_new});
                                end
                                if mod(iter,2) == 0 % half: get optimal hyp from previous depth kernels, with new ind pts and hyps for current depth kernel
                                    [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
                                    [lb_table(iter,i),gp_var_cell{iter,i}] = lbfunction(x,y,xu,gpcf_new,lik);
                                else % other half: use random init of hyp and ind pts
                                    [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
                                    [gpcf_new,lik_new] = reinitialise_kernel(gpcf_new,x,y);
                                    [lb_table(iter,i),gp_var_cell{iter,i}] = lbfunction(x,y,xu,gpcf_new,lik_new);
                                end

                            else % for 1/5 the iter, keep optimal ind pts and hyp from previous m
                                weights = 1e-10*ones(1,n); %weights for sampling
                                weights(idx_u)=1; %make sure samples idx_u are included
                                [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                                [lb_table(iter,i),gp_var_cell{iter,i}] = lbfunction(x,y,xu,gpcf_best,lik_best);
                            end
                        end
                        [~,ind] = max(lb_table(:,i));
                        idx_u = idx_cell{ind,i}; %indices of subset for best LB
                        %%% find ub for hyp from best LB
                        gp_var_best = gp_var_cell{ind,i};
                        gpcf_best = gp_var_best.cf{1};
                        lik_best = gp_var_best.lik;
                        ub_table(i) = ubfunction(x,y,gp_var_best,precond);
                    end
                    %%% gather best result for m=max(m_values), and store in kernel
                    [lb,ind] = max(lb_table(:,nm));
                    ub = ub_table(nm);
                    gp_var = gp_var_cell{ind,nm};
                    indices = idx_cell{ind,nm};
                    kernel = struct('key',key_new,'lb',lb,'ub',ub,'gp_var',gp_var,'indices',indices);
                    
                    %%% compare kernel with previous kernels
                    n_buffer = length(kernel_buffer);
                    if ub < kernel_top.lb % kernel interval strictly below top kernel interval
                        % ignore kernel
                    elseif lb < kernel_top.lb % kernel interval overlaps with top kernel interval, but has lower lb than top_kernel
                        [buffer_min_val, buffer_min_ind] = findmin(kernel_buffer);
                        if n_buffer < S % buffer not full
                            kernel_buffer(n_buffer+1) = kernel;
                        elseif lb > buffer_min_val % if kernel has higher lb than some kernel in buffer
                            kernel_buffer(buffer_min_ind) = kernel;
                        end
                    else % kernel.lb > kernel_top.lb
                        kernel_top = kernel;
                        %%% compare kernels in buffer to new kernel_top, and see if
                        %%% they should remain or be deleted
                        for buffer_ind = 1:length(kernel_buffer)
                            buffer_kernel = kernel_buffer(buffer_ind);
                            if buffer_kernel.ub < lb % if kernel in buffer has strictly lower interval than kernel_top
                                kernel_buffer(buffer_ind) = [];
                            end
                        end
                        n_buffer_new = length(kernel_buffer);
                        if n_buffer_new < S % if buffer is not full
                            kernel_buffer(n_buffer_new+1) = kernel;
                        else % buffer full, so replace the buffer kernel with the lowest lb
                            [~, buffer_min_ind] = findmin(kernel_buffer);
                            kernel_buffer(buffer_min_ind) = kernel;
                        end
                    end
                    fprintf([key_new ' done. lb=%4.2f, ub = %4.2f \n'],lb,ub);
                end
            end
        end
        kernel_new = findnew(kernel_buffer_old,kernel_buffer);
        end
    end
    kernel_top_history(length(kernel_top_history)+1) = kernel_top;
    kbh_length=length(kernel_buffer_history);
    kernel_buffer_history((kbh_length+1):(kbh_length+length(kernel_new)))=kernel_new;
    fprintf('depth %d done\n',depth);
end

end

function [min_val,min_ind] = findmin(buffer) % find min_ind, min_val of struct array of kernels
    n_buffer = length(buffer);
    if n_buffer == 0
        error('buffer is empty')
    end
    buffer_lb = zeros(1,n_buffer);
    for buffer_ind = 1:n_buffer % extract the lb of kernels in val into 
        buffer_lb(buffer_ind) = buffer(buffer_ind).lb;
    end
    [min_val,min_ind] = min(buffer_lb);
end

function kernel_new = findnew(kernel_buffer_old,kernel_buffer_new)
% find the new kernels that have been added to kernel_buffer_new from
% kernel_buffer old
    new_keys = cell(1,length(kernel_buffer_new));
    old_keys = cell(1,length(kernel_buffer_old));
    for ind = 1:length(kernel_buffer_new)
        new_keys{ind} = kernel_buffer_new(ind).key;
    end
    for ind = 1:length(kernel_buffer_old)
        old_keys{ind} = kernel_buffer_old(ind).key;
    end
    new_ind = find(~ismember(new_keys,old_keys)); % the indices of new_kernels in kernel_buffer_new
    kernel_new = kernel_buffer_new(new_ind); %logical indexing does not work for struct arrays
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

function [lb, gp_var] = lbfunction(x,y,xu,gpcf,lik) % get lb of bic
    n = size(x,1);
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', xu);
    gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp_var = gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
    if gp_var.lik.sigma2 > 1e-8
        [~,nll] = gp_e([],gp_var,x,y);
        ll = -nll;
        if ll > 0 
            lb =nan;
            return
        end
        p = length(gp_pak(gp_var)); % number of hyperparams
        lb = ll - p*log(n)/2;
    else lb = nan;
    end
end

function ub = ubfunction(x,y,gp_var,precond) % get ub of bic
warning('off','all');
if gp_var.lik.sigma2 > 1e-8
    xu = gp_var.X_u;
    m = size(xu,1);
    n = size(x,1);
    p = length(gp_pak(gp_var)); % number of hyperparams
    signal_var = gp_var.lik.sigma2;
    K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
    try
        L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
    catch
        L_mm=chol(K_mm+1e-8*eye(m));
    end
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
%     logprior = 0;
%     ncf = length(gp_var.cf); % number of hyp
%     for i=1:ncf
%         gpcf = gp_var.cf{i};
%         logprior = logprior + gpcf.fh.lp(gpcf);
%     end
%     lik = gp_var.lik;
%     logprior = logprior + lik.fh.lp(lik);
    
    ub = naive_nld + nip_ub - 0.5*n*log(2*pi) - p*log(n)/2;
else ub =nan;
end
end
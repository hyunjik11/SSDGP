function [kernel_dict, kernel_dict_debug] = kernel_tree(x,y,final_depth,num_iter,m_values,seed,fullgp,precond)
% function to get LB, UB and fullGP ml for each kernel in the space of
% compositional kernels up to depth=final_depth 
% with num_iter rand init for each kernel.
% gpcf_dict contains the kernels with optimal hyp for each of LB, UB, fullGP
% base_kernels is dictionary with string keys and gpcf values
% base_kernels should contain 3*ndims kernels where ndims is dim of x values
% fullgp is a boolean for whether fullgp optim should be done or not
% precond is the preconditioner used for PCG for ub - can take values from:
% 'None','Nystrom','FIC','PIC'
base_kernels = construct_base_kernels(x,y);

%%% initialise dictionary with key = kernel type, 
%%% value = cell of lb(+gp_var,indices of indpts),ub,fullgp(+gp) (size=6, order = lb, ub, fullGP)
kernel_dict = containers.Map('KeyType','char','ValueType','any');

%%% intialise dictionary of kernels at current depth with key = kernel type,
%%% value = cell of lb(+gp_var,indices of indpts),ub,fullgp(+gp) (size=6, order = lb, ub, fullGP)
kernel_dict_depth = containers.Map('KeyType','char','ValueType','any');

% debug contains lb(+gp_var,indices of indpts),ub,fullgp(+gp) for all iterations
kernel_dict_debug = containers.Map('KeyType','char','ValueType','any');

rng(seed);
[n,~] = size(x); % n is n_data, ndims is dimensionality of inputs
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
            gp_var_cell = cell(num_iter,nm); % stores lb gp_var for all iter
            if fullgp
                gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                gp_cell = cell(num_iter,1); % stores fullgp gp for all iter
            end
            idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
            idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
            [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
            for i = 1:nm
                m = m_values(i);
                parfor iter = 1:num_iter % change to parfor for parallel
                    %rng(iter);                    
                    % optim for lb
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
                    
                    % optim for fullgp
                    if fullgp && (i==1)
                        [gp_table(iter),gp_cell{iter}] = gpfunction(x,y,gpcf,lik);
                    end
                end
                [~,ind] = max(lb_table(:,i));
                idx_u = idx_cell{ind,i}; %indices of subset for best LB
                % find ub for hyp from best LB
                gp_var_best = gp_var_cell{ind,i};
                gpcf_best = gp_var_best.cf{1};
                lik_best = gp_var_best.lik;
                ub_table(i) = ubfunction(x,y,gp_var_best,precond);
            end
            % gather best results
            [lb,ind] = max(lb_table,[],1);
            lb_gp_var = cell(1,nm); lb_idx=cell(1,nm);
            for i=1:nm
                lb_gp_var{i} = gp_var_cell{ind(i),i};
                lb_idx{i} = idx_cell{ind(i),i};
            end
            
            if fullgp
                [ne,ind] = max(gp_table);
                gp = gp_cell{ind};
            end
            
            % store all data in debug_cell, value for kernel_dict_debug
            debug_cell = {};
            debug_cell{1} = lb_table; debug_cell{2} = gp_var_cell;
            debug_cell{3} = idx_cell; debug_cell{4} = ub_table; 
         
            % store best results in gpcf_dict_depth
            depth_cell = {};
            depth_cell{1} = lb; depth_cell{2} = lb_gp_var;
            depth_cell{3} = lb_idx; depth_cell{4} = ub_table; 
            if fullgp
                debug_cell{5} = gp_table; debug_cell{6} = gp_cell;
                depth_cell{5} = ne; depth_cell{6} = gp;
            end
            kernel_dict_debug(key) = debug_cell;
            kernel_dict_depth_new(key) = depth_cell;
            fprintf([key ' done \n']);
        end
        
    else %%% for depth > 1, branch factor = 6*ndims
        for key_ind = 1:length(kernel_dict_depth.keys)
            keys = kernel_dict_depth.keys; key = keys{key_ind};
            depth_cell = kernel_dict_depth(key);
            val = depth_cell{2}{nm}.cf{1}; % gpcf with hyp giving best lb for parent kernel with max m
            lik = depth_cell{2}{nm}.lik; % lik with hyp giving best lb for parent kernel with max m
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
                    gp_var_cell = cell(num_iter,nm); % stores lb gp_var for all iter
                    if fullgp
                        gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                        gp_cell = cell(num_iter,1); % stores fullgp gp for all iter
                    end
                    idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
                    idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
                    [gpcf_best,lik_best] = reinitialise_kernel(val,x,y); %temporary initialisation
                    for i = 1:nm
                        m = m_values(i);
                        parfor iter = 1:num_iter
                            %rng(iter);
                            % optim for lb
                            if i==1 || iter<=0.8*num_iter  
                                % for m_min, or for 4/5 of the iter, split
                                % into half and half:
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
                            
                            % optim for fullgp
                            if fullgp && (i==1)
                                [gp_table(iter),gp_cell{iter}] = gpfunction(x,y,gpcf_new,lik);
                            end
                        end
                        [~,ind] = max(lb_table(:,i));
                        idx_u = idx_cell{ind,i}; %indices of subset for best LB
                        % find ub for hyp from best LB
                        gp_var_best = gp_var_cell{ind,i};
                        gpcf_best = gp_var_best.cf{1};
                        lik_best = gp_var_best.lik;
                        ub_table(i) = ubfunction(x,y,gp_var_best,precond);
                    end
                    % gather best results
                    [lb,ind] = max(lb_table,[],1);
                    lb_gp_var = cell(1,nm); lb_idx=cell(1,nm);
                    for i=1:nm
                        lb_gp_var{i} = gp_var_cell{ind(i),i};
                        lb_idx{i} = idx_cell{ind(i),i};
                    end
                    
                    if fullgp
                        [ne,ind] = max(gp_table);
                        gp = gp_cell{ind};
                    end
                    
                    % store all data in debug_cell, value for kernel_dict_debug
                    debug_cell = {};
                    debug_cell{1} = lb_table; debug_cell{2} = gp_var_cell;
                    debug_cell{3} = idx_cell; debug_cell{4} = ub_table;
                    
                    % store best results in gpcf_dict_depth
                    depth_cell = {};
                    depth_cell{1} = lb; depth_cell{2} = lb_gp_var;
                    depth_cell{3} = lb_idx; depth_cell{4} = ub_table;
                    if fullgp
                        debug_cell{5} = gp_table; debug_cell{6} = gp_cell;
                        depth_cell{5} = ne; depth_cell{6} = gp;
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
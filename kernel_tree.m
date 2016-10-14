function [kernel_dict, kernel_dict_debug] = kernel_tree(x,y,base_kernels,final_depth,num_iter,m_values,seed,fullgp)
% function to get LB, UB and fullGP ml for each kernel in the space of
% compositional kernels up to depth=final_depth 
% with numiter rand init for each kernel.
% gpcf_dict contains the kernels with optimal hyp for each of LB, UB, fullGP
% base_kernels is dictionary with string keys and gpcf values
% fullgp is a boolean for whether fullgp optim should be done or not

%%% initialise dictionary with key = kernel type, 
%%% value = cell of lb(+gpcf,lik),ub(+gpcf,lik),fullgp(+gpcf,lik) (size=9, order = lb, ub, fullGP)
kernel_dict = containers.Map('KeyType','char','ValueType','any');

%%% intialise dictionary of kernels at current depth with key = kernel type,
%%% value = cell of lb(+gpcf,lik),ub(+gpcf,lik),fullgp(+gpcf,lik) (size=9, order = lb, ub, fullGP)
kernel_dict_depth = containers.Map('KeyType','char','ValueType','any');

% debug contains lb(+gpcf,lik), ub(+gpcf,lik), fullgp(+gpcf,lik) for all iterations
kernel_dict_debug = containers.Map('KeyType','char','ValueType','any');

rng(seed);
[n,ndims] = size(x); % n is n_data, ndims is dimensionality of inputs
%%% TODO: fix algorithm for ndims > 1 (multidim inputs e.g. concrete)

nm = length(m_values);

%%% kernel search
for depth = 1:final_depth
    kernel_dict_depth_new = containers.Map('KeyType','char','ValueType','any');
    if depth == 1 %%% for first depth, branch factor = 3
        for key_ind = 1:length(base_kernels.keys)
            keys = base_kernels.keys; key = keys{key_ind};
            val = base_kernels(key);
            lb_table = zeros(num_iter,nm); % stores lb for all iter
            ub_table = zeros(num_iter,nm); % stores ub for all iter
            lb_gpcf_cell = cell(num_iter,nm); % stores lb gpcf for all iter
            ub_gpcf_cell = cell(num_iter,nm); % stores ub gpcf for all iter
            lb_lik_cell = cell(num_iter,nm); % stores lb lik for all iter
            ub_lik_cell = cell(num_iter,nm); % stores ub lik for all iter
            if fullgp
                gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                gp_gpcf_cell = cell(num_iter,1); % stores fullgp gpcf for all iter
                gp_lik_cell = cell(num_iter,1); % stores fullgp lik for all iter
            end
            idx_cell = cell(num_iter,1); % stores indices for ind pts for all iter
            idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
            for i = 1:nm
                m = m_values(i);                
                parfor iter = 1:num_iter
                    weights = 1e-10*ones(1,n); %weights for sampling
                    weights(idx_u)=1; %make sure samples idx_u are included
                    [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                    
                    % optim for lb
                    [gpcf, lik] = reinitialise_kernel(val,x,y);
                    [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf,lik);
                    
                    % optim for ub
                    [gpcf, lik] = reinitialise_kernel(val,x,y);
                    [ub_table(iter,i),ub_gpcf_cell{iter,i},ub_lik_cell{iter,i}] = ubfunction(x,y,xu,gpcf,lik);
                    
                    % optim for fullgp
                    if fullgp && (i==1)
                        [gp_table(iter),gp_gpcf_cell{iter},gp_lik_cell{iter}] = gpfunction(x,y,gpcf,lik);
                    end
                end
                [~,ind] = max(lb_table(:,i));
                idx_u = idx_cell{ind,i}; %indices of subset for best LB
            end
            % gather best results
            [lb,ind] = max(lb_table,[],1);
            lb_gpcf = cell(1,nm); lb_lik = cell(1,nm);
            for i=1:nm
                lb_gpcf{i} = lb_gpcf_cell{ind(i),i};
                lb_lik{i} = lb_lik_cell{ind(i),i};
            end
            
            [ub,ind] = max(ub_table,[],1);
            ub_gpcf = cell(1,nm); ub_lik = cell(1,nm);
            for i=1:nm
                ub_gpcf{i} = ub_gpcf_cell{ind(i),i};
                ub_lik{i} = ub_lik_cell{ind(i),i};
            end
            
            if fullgp
                [ne,ind] = max(gp_table);
                gp_gpcf = gp_gpcf_cell{ind};
                gp_lik = gp_lik_cell{ind};
            end
            
            % store all data in debug_cell, value for kernel_dict_debug
            debug_cell = {};
            debug_cell{1} = lb_table; debug_cell{2} = lb_gpcf_cell; debug_cell{3} = lb_lik_cell;
            debug_cell{4} = ub_table; debug_cell{5} = ub_gpcf_cell; debug_cell{6} = ub_lik_cell;
         
            % store best results in gpcf_dict_depth
            depth_cell = {};
            depth_cell{1} = lb; depth_cell{2} = lb_gpcf; depth_cell{3} = lb_lik;
            depth_cell{4} = ub; depth_cell{5} = ub_gpcf; depth_cell{6} = ub_lik;
            if fullgp
                debug_cell{7} = gp_table; debug_cell{8} = gp_gpcf_cell; debug_cell{9} = gp_lik_cell;
                depth_cell{7} = ne; depth_cell{8} = gp_gpcf; depth_cell{9} = gp_lik;
            end
            kernel_dict_debug(key) = debug_cell;
            kernel_dict_depth_new(key) = depth_cell;
            fprintf([key ' done \n']);
        end
        
    else %%% for depth > 1, branch factor = 6
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
                    ub_table = zeros(num_iter,nm); % stores ub for all iter
                    lb_gpcf_cell = cell(num_iter,nm); % stores lb gpcf for all iter
                    ub_gpcf_cell = cell(num_iter,nm); % stores ub gpcf for all iter
                    lb_lik_cell = cell(num_iter,nm); % stores lb lik for all iter
                    ub_lik_cell = cell(num_iter,nm); % stores ub lik for all iter
                    if fullgp
                        gp_table = zeros(num_iter,1); % stores fullgp ml for all iter
                        gp_gpcf_cell = cell(num_iter,1); % stores fullgp gpcf for all iter
                        gp_lik_cell = cell(num_iter,1); % stores fullgp lik for all iter
                    end
                    idx_cell = cell(num_iter,1); % stores indices for ind pts for all iter
                    idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
                    for i = 1:nm
                        m = m_values(i);
                        parfor iter = 1:num_iter
                            weights = 1e-10*ones(1,n); %weights for sampling
                            weights(idx_u)=1; %make sure samples idx_u are included
                            [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
                            [val_base_new,~] = reinitialise_kernel(val_base,x,y);
                            if comp==0 % kernel in previous depth + base kernel
                                gpcf = gpcf_sum('cf',{val,val_base_new});
                            else % kernel in previous depth * base kernel
                                gpcf = gpcf_prod('cf',{val,val_base_new});
                            end
                            % optim for lb
                            [lb_table(iter,i),lb_gpcf_cell{iter,i},lb_lik_cell{iter,i}] = lbfunction(x,y,xu,gpcf,lik);
                            % optim for ub
                            [ub_table(iter,i),ub_gpcf_cell{iter,i},ub_lik_cell{iter,i}] = ubfunction(x,y,xu,gpcf,lik);
                            % optim for fullgp
                            if fullgp && (i==1)
                                [gp_table(iter),gp_gpcf_cell{iter},gp_lik_cell{iter}] = gpfunction(x,y,gpcf,lik);
                            end
                        end
                        [~,ind] = max(lb_table(:,i));
                        idx_u = idx_cell{ind,i}; %indices of subset for best LB
                    end
                    % gather best results
                    [lb,ind] = max(lb_table,[],1);
                    lb_gpcf = cell(1,nm); lb_lik = cell(1,nm);
                    for i=1:nm
                        lb_gpcf{i} = lb_gpcf_cell{ind(i),i};
                        lb_lik{i} = lb_lik_cell{ind(i),i};
                    end
                    
                    [ub,ind] = max(ub_table,[],1);
                    ub_gpcf = cell(1,nm); ub_lik = cell(1,nm);
                    for i=1:nm
                        ub_gpcf{i} = ub_gpcf_cell{ind(i),i};
                        ub_lik{i} = ub_lik_cell{ind(i),i};
                    end
                    
                    if fullgp
                        [ne,ind] = max(gp_table);
                        gp_gpcf = gp_gpcf_cell{ind};
                        gp_lik = gp_lik_cell{ind};
                    end
                    
                    % store all data in debug_cell, value for kernel_dict_debug
                    debug_cell = {};
                    debug_cell{1} = lb_table; debug_cell{2} = lb_gpcf_cell; debug_cell{3} = lb_lik_cell;
                    debug_cell{4} = ub_table; debug_cell{5} = ub_gpcf_cell; debug_cell{6} = ub_lik_cell;
                    
                    % store best results in gpcf_dict_depth
                    depth_cell = {};
                    depth_cell{1} = lb; depth_cell{2} = lb_gpcf; depth_cell{3} = lb_lik;
                    depth_cell{4} = ub; depth_cell{5} = ub_gpcf; depth_cell{6} = ub_lik;
                    if fullgp
                        debug_cell{7} = gp_table; debug_cell{8} = gp_gpcf_cell; debug_cell{9} = gp_lik_cell;
                        depth_cell{7} = ne; depth_cell{8} = gp_gpcf; depth_cell{9} = gp_lik;
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

function [gpcf, lik] = reinitialise_kernel(gpcf, x, y)
     % function to reinitialise gpcf of base kernels and lik
     ndims = size(x,2);
     switch gpcf.type
         case 'gpcf_sexp'
             gpcf = se_init(x,y,ndims);
         case 'gpcf_linear'
             gpcf = lin_init(ndims);
         case 'gpcf_periodic'
             gpcf = per_init(x,y,ndims);
         otherwise error('gpcf of invalid type - not in base_kernels');
     end
     lik = lik_init(y);
end

function [lb, gpcf, lik] = lbfunction(x,y,xu,gpcf,lik)
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', xu);
    gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
    lb = - gp_e([],gp_var,x,y);
    gpcf = gp_var.cf{1};
    lik = gp_var.lik;
end

function [ub, gpcf, lik] = ubfunction(x,y,xu,gpcf,lik)
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', xu);
    gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    [gp_var,val]=approx_ub(gp_var,x,y,opt);
    ub = -val;
    gpcf = gp_var.cf{1};
    lik = gp_var.lik;
end

function [ne, gpcf, lik] = gpfunction(x,y,gpcf,lik)
    gp=gp_set('lik',lik,'cf',gpcf);
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp = gp_optim(gp,x,y,'opt',opt,'optimf',@fminscg);
    ne = - gp_e([],gp,x,y);
    gpcf = gp.cf{1};
    lik = gp.lik;
end
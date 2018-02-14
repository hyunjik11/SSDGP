function [kernel_buffer,kernel_buffer_history, kernel_top, kernel_top_history] = skc_parallel(x,y,final_depth,num_iter,m,seed,S,precond,string)
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

start = tic;

for depth = 1:final_depth
    if depth == 1
        keys = base_kernels.keys;
        lk = length(keys);
        full_lb_table = zeros(num_iter*lk,1);
        full_gp_var_cell = cell(num_iter*lk,1);
        full_idx_cell = cell(num_iter*lk,1);
        parfor kernel_ind = 1:(num_iter*lk)
            key_ind = floor((kernel_ind-1)/num_iter)+1; % kernel_ind: 1-num_iter -> key_ind: 1 and so on
            key = keys{key_ind}; % name of kernel
            val = base_kernels(key); % gpcf in base kernel
            
            %%% optim for lb
            rng(kernel_ind);
            [xu,full_idx_cell{kernel_ind}] = datasample(x,m,1,'Replace',false);
            [gpcf, lik] = reinitialise_kernel(val,x,y);
            [full_lb_table(kernel_ind),full_gp_var_cell{kernel_ind}] = lbfunction(x,y,xu,gpcf,lik);
        end
        
        for key_ind = 1:lk
            key = keys{key_ind}; % name of kernel
            [lb,ind] = max(full_lb_table(((key_ind-1)*num_iter+1):(key_ind*num_iter)));
            indices = full_idx_cell{(key_ind-1)*num_iter+ind}; %indices of subset for best LB
            gp_var_best = full_gp_var_cell{(key_ind-1)*num_iter+ind};    
            
            %%% find ub for hyp from best LB
            ub = ubfunction(x,y,gp_var_best,precond);
            
            kernel = struct('key',key,'lb',lb,'ub',ub,'gp_var',gp_var_best,'indices',indices);

            %%% compare kernel with previous kernels
            n_buffer = length(kernel_buffer);
            if n_buffer == 0 % buffer is empty
                kernel_buffer(1) = kernel; kernel_top = kernel;
%             elseif ub < kernel_top.lb % kernel interval strictly below top kernel interval
%                 % ignore kernel
            elseif lb < kernel_top.lb % kernel has lower lb than top kernel
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
%                 del_ind=[]; % indices of kernels to be deleted
%                 for buffer_ind = 1:length(kernel_buffer)
%                     buffer_kernel = kernel_buffer(buffer_ind);
%                     if buffer_kernel.ub < lb % if kernel in buffer has strictly lower interval than kernel_top
%                         del_ind(length(del_ind)+1)=buffer_ind;
%                     end
%                 end
%                 kernel_buffer(del_ind)=[]; % delete kernels
                n_buffer_new = n_buffer;
                if n_buffer_new < S % if buffer is not full
                    kernel_buffer(n_buffer_new+1) = kernel;
                else % buffer full, so replace the buffer kernel with the lowest lb
                    [~, buffer_min_ind] = findmin(kernel_buffer);
                    kernel_buffer(buffer_min_ind) = kernel;
                end
            end
            fprintf([key ' : lb = %4.2f, ub = %4.2f \n'],lb,ub);
        end
        kernel_new = kernel_buffer;
    else % if depth > 1
        if isempty(kernel_new) % no new kernels found in search
            return
        else
            kernel_buffer_old = kernel_buffer; % need for comparing with kernel_buffer after search at current depth
            lkn = length(kernel_new); lbk = length(base_kernels.keys);
            lk = lkn*lbk*2;
            full_lb_table = zeros(num_iter*lk,1);
            full_gp_var_cell = cell(num_iter*lk,1);
            full_idx_cell = cell(num_iter*lk,1);
            full_key_cell = cell(num_iter*lk,1);
            parfor kernel_ind = 1:(num_iter*lk)
                parent_ind = floor((kernel_ind-1)/(num_iter*lbk*2))+1; % kernel_ind: 1-num_iter*lbk*2 -> parent_ind: 1 and so on
                key = kernel_new(parent_ind).key;
                val = kernel_new(parent_ind).gp_var.cf{1};
                lik = kernel_new(parent_ind).gp_var.lik;
                base_key_ind = floor((kernel_ind-(parent_ind-1)*lbk*num_iter*2-1)/(num_iter*2))+1;
                base_keys = base_kernels.keys; key_base = base_keys{base_key_ind};
                val_base = base_kernels(key_base);
                comp = floor((kernel_ind-(parent_ind-1)*lbk*num_iter*2-(base_key_ind-1)*num_iter*2-1)/num_iter)+1;
                if comp == 1 % kernel in previous depth + base kernel
                    key_new = ['(' key ')+' key_base];
                else % comp =2. kernel in previous depth * base kernel
                    key_new = ['(' key ')*' key_base];
                end
                full_key_cell{kernel_ind} = key_new;
                %%% optim for lb:
                rng(kernel_ind);
                [val_base_new,~] = reinitialise_kernel(val_base,x,y);
                if comp == 1 % kernel in previous depth + base kernel
                    gpcf_new = gpcf_sum('cf',{val,val_base_new});
                else % kernel in previous depth * base kernel
                    gpcf_new = gpcf_prod('cf',{val,val_base_new});
                end
                if mod(kernel_ind,2) == 0 % half: get optimal hyp from previous depth kernels, with new ind pts and hyps for current depth kernel
                    [xu,full_idx_cell{kernel_ind}] = datasample(x,m,1,'Replace',false);
                    [full_lb_table(kernel_ind),full_gp_var_cell{kernel_ind}] = lbfunction(x,y,xu,gpcf_new,lik);
                else % other half: use random init of hyp and ind pts
                    [xu,full_idx_cell{kernel_ind}] = datasample(x,m,1,'Replace',false);
                    [gpcf_new,lik_new] = reinitialise_kernel(gpcf_new,x,y);
                    [full_lb_table(kernel_ind),full_gp_var_cell{kernel_ind}] = lbfunction(x,y,xu,gpcf_new,lik_new);
                end
                %fprintf('kernel_ind:%d, parent_ind:%d, base_key_ind:%d, comp:%d \n',kernel_ind,parent_ind,base_key_ind,comp);
            end
            
            for key_ind = 1:lk
                key = full_key_cell{(key_ind-1)*num_iter+1};
                [lb,ind] = max(full_lb_table(((key_ind-1)*num_iter+1):(key_ind*num_iter)));
                indices = full_idx_cell{(key_ind-1)*num_iter+ind};
                gp_var_best = full_gp_var_cell{(key_ind-1)*num_iter+ind};
                
                %%% find ub for hyp from best lb
                ub = ubfunction(x,y,gp_var_best,precond);

                kernel = struct('key',key,'lb',lb,'ub',ub,'gp_var',gp_var_best,'indices',indices);
                
                %%% compare kernel with previous kernels
                n_buffer = length(kernel_buffer);
%                 if ub < kernel_top.lb % kernel interval strictly below top kernel interval
%                     % ignore kernel
%                 elseif lb < kernel_top.lb % kernel interval overlaps with top kernel interval, but has lower lb than top_kernel
                if lb < kernel_top.lb % kernel interval overlaps with top kernel interval, but has lower lb than top_kernel
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
%                     del_ind=[]; % indices of kernels to be deleted
%                     for buffer_ind = 1:length(kernel_buffer)
%                         buffer_kernel = kernel_buffer(buffer_ind);
%                         if buffer_kernel.ub < lb % if kernel in buffer has strictly lower interval than kernel_top
%                             del_ind(length(del_ind)+1)=buffer_ind;
%                         end
%                     end
%                     kernel_buffer(del_ind)=[]; % delete kernels
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
        end
        kernel_new = findnew(kernel_buffer_old,kernel_buffer);
    end
    kernel_top_history(length(kernel_top_history)+1) = kernel_top;
    kbh_length=length(kernel_buffer_history);
    kernel_buffer_history((kbh_length+1):(kbh_length+length(kernel_new)))=kernel_new;
    fprintf('depth %d done\n',depth);
    fprintf('%f elapsed since start\n',toc(start))
    save(string,'kernel_buffer', 'kernel_buffer_history', 'kernel_top', 'kernel_top_history');
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


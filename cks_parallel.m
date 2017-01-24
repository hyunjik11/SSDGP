function [kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = cks_parallel(x,y,final_depth,num_iter,seed,S)
% function to carry out compositional kernel search on inputs x, outputs y
% up to depth = final_depth
% with num_iter rand inits for each kernel
% kernel_buffer is a struct array of size S which contains the top S kernels.
% kernel_top is the struct for the kernel found by skd.
% kernel_top_history is a struct array of kernels that contains the kernel
% selected at each depth.
% Each kernel is a struct with fields: key, BIC, gp
base_kernels = construct_base_kernels(x,y); % create a dictionary of base kernels

%%% initialise kernel_buffer
kernel_buffer = struct('key',{},'bic',{},'gp',{});

%%% initialise kernel_buffer_history
kernel_buffer_history = struct('key',{},'bic',{},'gp',{});

%%% initialise kernel_top
kernel_top = struct('key',[],'bic',[],'gp',[]);

%%% initialise kernel_top_history
kernel_top_history = struct('key',{},'bic',{},'gp',{});

%%% initialise kernel_new, the cell array of kernels newly added to
%%% kernel_buffer at each depth - for the following depth, the three will
%%% only grow on these kernels
kernel_new = struct('key',{},'bic',{},'gp',{});

rng(seed);
[n,~] = size(x); % n is n_data

for depth = 1:final_depth
    if depth == 1
        keys = base_kernels.keys;
        lk = length(keys);
        full_bic_table = zeros(num_iter*lk,1);
        full_gp_cell = cell(num_iter*lk,1);
        parfor kernel_ind = 1:(num_iter*lk)
            key_ind = floor((kernel_ind-1)/num_iter)+1; % kernel_ind: 1-num_iter -> key_ind: 1 and so on
            key = keys{key_ind}; % name of kernel
            val = base_kernels(key); % gpcf in base kernel
            %%% optim for gp
            rng(kernel_ind);
            [gpcf, lik] = reinitialise_kernel(val,x,y);
            [full_bic_table(kernel_ind),full_gp_cell{kernel_ind}] = gpfunction(x,y,gpcf,lik);
        end
        
        for key_ind = 1:lk
            key = keys{key_ind}; % name of kernel
            [bic,ind] = max(full_bic_table(((key_ind-1)*num_iter+1):(key_ind*num_iter)));
            gp = full_gp_cell{(key_ind-1)*num_iter+ind};
            kernel = struct('key',key,'bic',bic,'gp',gp);
            
            %%% compare kernel with previous kernels
            n_buffer = length(kernel_buffer);
            if n_buffer == 0 % buffer is empty
                kernel_buffer(1) = kernel; kernel_top = kernel;
            else
                if bic > kernel_top.bic
                    kernel_top = kernel;
                end
                if n_buffer < S % buffer not full
                    kernel_buffer(n_buffer+1) = kernel;
                else
                    [buffer_min_val, buffer_min_ind] = findmin(kernel_buffer);
                    if bic > buffer_min_val % if kernel has higher bic than some kernel in buffer
                        kernel_buffer(buffer_min_ind) = kernel; % replace that kernel
                    end
                end
            end
            fprintf([key ' done. bic=%4.2f \n'],bic);
        end
        kernel_new = kernel_buffer;
    else % if depth > 1
        if isempty(kernel_new) % no new kernels found in search
            return
        else
            kernel_buffer_old = kernel_buffer; % need for comparing with kernel_buffer after search at current depth
            lkn = length(kernel_new); lbk = length(base_kernels.keys);
            lk = lkn*lbk*2;
            full_bic_table = zeros(num_iter*lk,1);
            full_gp_cell = cell(num_iter*lk,1);
            full_key_cell = cell(num_iter*lk,1);
            parfor kernel_ind = 1:(num_iter*lk)
                parent_ind = floor((kernel_ind-1)/(num_iter*lbk*2))+1; % kernel_ind: 1-num_iter*lbk*2 -> parent_ind: 1 and so on
                key = kernel_new(parent_ind).key;
                val = kernel_new(parent_ind).gp.cf{1};
                lik = kernel_new(parent_ind).gp.lik;
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
                %%% optim for gp
                rng(kernel_ind);
                [val_base_new,~] = reinitialise_kernel(val_base,x,y);
                if comp==1 % kernel in previous depth + base kernel
                    gpcf_new = gpcf_sum('cf',{val,val_base_new});
                else % kernel in previous depth * base kernel
                    gpcf_new = gpcf_prod('cf',{val,val_base_new});
                end
                if mod(kernel_ind,2) == 0 % half: get optimal hyp from previous depth kernels and use new hyps for current depth kernel
                    [full_bic_table(kernel_ind),full_gp_cell{kernel_ind}] = gpfunction(x,y,gpcf_new,lik);
                else % other half: use random init of hyps
                    [gpcf_new, lik_new] = reinitialise_kernel(gpcf_new,x,y);
                    [full_bic_table(kernel_ind),full_gp_cell{kernel_ind}] = gpfunction(x,y,gpcf_new,lik_new);
                end
            end
            
            for key_ind = 1:lk
                key = full_key_cell{(key_ind-1)*num_iter+1};
                [bic,ind] = max(full_bic_table(((key_ind-1)*num_iter+1):(key_ind*num_iter)));
                gp = full_gp_cell{(key_ind-1)*num_iter+ind};
                kernel = struct('key',key,'bic',bic,'gp',gp);
                
                %%% compare kernel with previous kernels
                n_buffer = length(kernel_buffer);
                if bic > kernel_top.bic
                    kernel_top = kernel;
                end
                if n_buffer < S % buffer not full
                    kernel_buffer(n_buffer+1) = kernel;
                else
                    [buffer_min_val, buffer_min_ind] = findmin(kernel_buffer);
                    if bic > buffer_min_val % if kernel has higher bic than some kernel in buffer
                        kernel_buffer(buffer_min_ind) = kernel; % replace that kernel
                    end
                end
                fprintf([key ' done. bic=%4.2f \n'],bic);
            end
        end
        kernel_new = findnew(kernel_buffer_old,kernel_buffer);
    end
    kernel_top_history(length(kernel_top_history)+1) = kernel_top;
    kbh_length=length(kernel_buffer_history);
    kernel_buffer_history((kbh_length+1):(kbh_length+length(kernel_new)))=kernel_new;
    fprintf('depth %d done\n',depth);
end

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

function [min_val,min_ind] = findmin(buffer) % find min_ind, min_val of struct array of gp kernels
    n_buffer = length(buffer);
    if n_buffer == 0
        error('buffer is empty')
    end
    buffer_bic = zeros(1,n_buffer);
    for buffer_ind = 1:n_buffer % extract the lb of kernels in val into 
        buffer_bic(buffer_ind) = buffer(buffer_ind).bic;
    end
    [min_val,min_ind] = min(buffer_bic);
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
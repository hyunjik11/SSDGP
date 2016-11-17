function [gpcf, lik] = reinitialise_kernel(gpcf, x, y)
    % function to reinitialise gpcf of composite kernel and lik
    if strcmp(gpcf.type,'gpcf_sum') || strcmp(gpcf.type,'gpcf_prod')
        gpcf.cf{1} = reinitialise_kernel(gpcf.cf{1},x,y);
        gpcf.cf{2} = reinitialise_kernel(gpcf.cf{2},x,y);
    else % if gpcf is one of the base kernels
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
    end
    lik = lik_init(y);
end
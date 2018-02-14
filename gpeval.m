function [bic, gp] = gpeval(x,y,gpcf,lik)
    [n,~] = size(x); % n is n_data
    gp=gp_set('lik',lik,'cf',gpcf);
    p = length(gp_pak(gp)); % p is number of hyperparams
    [~, nll] = gp_e([],gp,x,y);
    bic = -nll - p*log(n)/2;
end
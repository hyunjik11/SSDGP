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
        p = length(gp_pak(gp_var)); % number of hyperparams
        lb = ll - p*log(n)/2;
    else lb = nan;
    end
end
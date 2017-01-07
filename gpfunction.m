function [bic, gp] = gpfunction(x,y,gpcf,lik)
    [n,~] = size(x); % n is n_data
    gp=gp_set('lik',lik,'cf',gpcf);
    p = length(gp_pak(gp)); % p is number of hyperparams
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    warning('off','all');
    gp = gp_optim(gp,x,y,'opt',opt,'optimf',@fminscg);
    % p = length(gp_pak(gp));
    [~, nll] = gp_e([],gp,x,y);
    bic = -nll - p*log(n)/2;
end
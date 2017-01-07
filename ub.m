function [bic_ub,gp_var_new] = ub(gp_var,x,y)
% optimizer to solve max_theta {0.5*log(det(K_nyst+sigma_sq*eye(n))+...
% 0.5*y'*inv(K_pic + sigma_sq*eye(n))*y - logp(theta) +n/2*log(2*pi)}
% val = -ve log-lik approx (the objective that is minimised)
    warning('off','all');
    n = size(x,1);
    w = gp_pak(gp_var);
    p = length(w); % number of hyperparams
    myfn = @(ww) ub_grad(ww,gp_var,x,y);
    optdefault=struct('GradObj','on','LargeScale','off');
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    opt=setOpt(optdefault,opt);
    [w_new, val] = fminscg(myfn,w,opt);
    bic_ub = -val - p*log(n)/2;
    gp_var_new = gp_unpak(gp_var,w_new);
end

function opt=setOpt(optdefault, opt)
% Set default options
opttmp=optimset(optdefault,opt);

% Set some additional options for @fminscg
if isfield(opt,'lambda')
    opttmp.lambda=opt.lambda;
end
if isfield(opt,'lambdalim')
    opttmp.lambdalim=opt.lambdalim;
end
opt=opttmp;
end
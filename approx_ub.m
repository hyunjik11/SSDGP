function [gp_var_new,val] = approx_ub(gp_var,x,y,opt)
% optimizer to solve max_theta {0.5*log(det(K_nyst+sigma_sq*eye(n))+...
% 0.5*y'*inv(K_pic + sigma_sq*eye(n))*y - logp(theta) +n/2*log(2*pi)}
% val = -ve log-lik approx (the objective that is minimised)
    warning('off','all');
    w = gp_pak(gp_var);
    myfn = @(ww) approx_ub_grad(ww,gp_var,x,y);
    optdefault=struct('GradObj','on','LargeScale','off');
    opt=setOpt(optdefault,opt);
    [w_new, val] = fminscg(myfn,w,opt);
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
function gpcf=per_init(x,y)
    std_x = std(x);
    std_y = std(y);
    n=length(x);
    pl=prior_gaussian('mu',std_x,'s2',0.25);
    minp = log(10*(max(x)-min(x))/n);
    plper=prior_loggaussian('mu',minp-2);
    gpcf = gpcf_periodic('lengthScale',exp(minp+TruncatedGaussian(-1/2,[0,Inf])),...
        'period',exp(minp+TruncatedGaussian(-1/2,[0,Inf])),'magnSigma2',...
        std_y*exp(randn()/2),'lengthScale_sexp_prior',prior_t(),...
        'period_prior',plper);
end